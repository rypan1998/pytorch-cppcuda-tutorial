# pytorch-cppcuda-tutorial
tutorial for writing custom pytorch cpp+cuda kernel, applied on volume rendering (NeRF)

tutorial playlist: https://www.youtube.com/watch?v=l_Rpk6CRJYI&list=PLDV2CyUo4q-LKuiNltBqCKdO9GH4SS_ec&ab_channel=AI%E8%91%B5


CUDA explanation in video 2: https://nyu-cds.github.io/python-gpu/02-cuda/

C++ API in video 3: https://pytorch.org/cppdocs/

kernel launching: https://pytorch.org/tutorials/advanced/cpp_extension.html

## 补充：如何在Colab上运行（2023年12月3日）

```python
!git clone https://github.com/rypan1998/pytorch-cppcuda-tutorial
!cd pytorch-cppcuda-tutorial && pip install . && python test.py
```