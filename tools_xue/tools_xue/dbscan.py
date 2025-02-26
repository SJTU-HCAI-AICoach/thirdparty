from torch.autograd import Function
import tools_xue_cuda


class DBSCAN(Function):
    @staticmethod
    def forward(ctx, xyz, eps, min_point):
        out = tools_xue_cuda.dbscan_cuda(xyz, eps, min_point)
        ctx.mark_non_differentiable(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()
        
dbscan_cuda = DBSCAN.apply