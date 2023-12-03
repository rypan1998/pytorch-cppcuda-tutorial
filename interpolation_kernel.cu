#include <torch/extension.h>


template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    // 如果超过输出范围，则跳过不能算
    if (n>=feats.size(0) || f>=feats.size(2)) return;

    // point -1~1. 需要正规化到 [0,1] 之间
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;
    
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;
    // 保存计算结果
    feat_interp[n][f] = (1-u)*(a*feats[n][0][f] +
                               b*feats[n][1][f] +
                               c*feats[n][2][f] +
                               d*feats[n][3][f]) + 
                            u*(a*feats[n][4][f] +
                               b*feats[n][5][f] +
                               c*feats[n][6][f] +
                               d*feats[n][7][f]);
}


torch::Tensor trilinear_fw_cu(
    const torch::Tensor feats,
    const torch::Tensor points
){
    const int N = feats.size(0), F = feats.size(2);
    
    // 定义 feat_interp 作为输出，数据格式、GPU等属性必须要和 feats 保持一致，因此这里直接用 feats.options() 作为所有数据属性的输入即可
    torch::Tensor feat_interp = torch::empty({N, F}, feats.options());
    // 知识拓展：如果没法用 options 而是需要自定义写法（比如最后的输出为整型）？
    // torch::empty({N,F}, torch::dtype(torch::kInt32).device(feats.device));
    // 解释：kInt32 为整型，直接紧跟它之后通过 device 指定 gpu 即可

    const dim3 threads(16, 16);// 这里N和F是可并行的，共两个次元（256不会出错）
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    // 建议调用该Kernel的第二个参数为当前函数名，方便报错debug
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", 
    ([&] {
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return feat_interp;
}


template <typename scalar_t>
__global__ void trilinear_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dfeat_interp,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,
    torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> dL_dfeats
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int f = blockIdx.y * blockDim.y + threadIdx.y;

    if (n>=feats.size(0) || f>=feats.size(2)) return;

    // point -1~1
    const scalar_t u = (points[n][0]+1)/2;
    const scalar_t v = (points[n][1]+1)/2;
    const scalar_t w = (points[n][2]+1)/2;
    
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;

    // 八个格点，每个格点的特征值都需要更新
    dL_dfeats[n][0][f] = (1-u)*a*dL_dfeat_interp[n][f];
    dL_dfeats[n][1][f] = (1-u)*b*dL_dfeat_interp[n][f];
    dL_dfeats[n][2][f] = (1-u)*c*dL_dfeat_interp[n][f];
    dL_dfeats[n][3][f] = (1-u)*d*dL_dfeat_interp[n][f];
    dL_dfeats[n][4][f] = u*a*dL_dfeat_interp[n][f];
    dL_dfeats[n][5][f] = u*b*dL_dfeat_interp[n][f];
    dL_dfeats[n][6][f] = u*c*dL_dfeat_interp[n][f];
    dL_dfeats[n][7][f] = u*d*dL_dfeat_interp[n][f];
}

// 把计算得到的 Loss 通过求导的方式，反向传播对 F 进行更新
torch::Tensor trilinear_bw_cu(
    const torch::Tensor dL_dfeat_interp,// Loss对feat_interp的偏导，已知量
    const torch::Tensor feats,
    const torch::Tensor points
){
    const int N = feats.size(0), F = feats.size(2);
    
    // 输出结果：Loss对各顶点特征值的偏导。特征值的维度是 [N,8,F] 那么偏导值也是这个维度
    torch::Tensor dL_dfeats = torch::empty({N, 8, F}, feats.options());

    const dim3 threads(16, 16);
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_bw_cu", 
    ([&] {
        trilinear_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dfeat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),// 由于 feat_interp 的维度是 [N,F]，所以求偏导后的结果也是这个维度，所以是 2
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dfeats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dfeats;
}
