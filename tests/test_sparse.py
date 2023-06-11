import torch
from torch_sparse import SparseTensor


mat = torch.rand(30, 30, 20)
mat = torch.dropout(mat, train=True, p=0.7)


row_, col_, hop_ = mat.nonzero(as_tuple=True)
val_ = mat[mat != 0]


row, col, hop, val = [], [], [], []

for i in range(mat.size(-1)):
    a = mat[:, :, i]
    row_t, col_t, val_t = SparseTensor.from_dense(a, has_value=True).coo()
    row.append(row_t)
    col.append(col_t)
    val.append(val_t)
    hop.append(torch.ones_like(val_t) * i)



for i in range(mat.size(-1)):
    row_same = (row_[hop_==i] == row[i]).any()
    col_same = (col_[hop_==i] == col[i]).any()
    val_same = (val_[hop_==i] == val[i]).any()

    print(f"row={row_same}, col={col_same}, val={val_same}, hop={i}")



breakpoint()

