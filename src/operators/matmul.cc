#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
                

        // 检查输入数量

        if (inputs.size() != 2) {

            return std::nullopt;

        }

        

        Shape A_dims = inputs[0]->getDims();

        auto A_rank = inputs[0]->getRank();

        Shape B_dims = inputs[1]->getDims();

        auto B_rank = inputs[1]->getRank();

        

        IT_ASSERT(A_rank >= 2 && B_rank >= 2);

        

        // 获取转置参数

        bool transA = this->getTransA();

        bool transB = this->getTransB();

        

        // 计算A和B在矩阵乘法中使用的维度

        int A_M, A_K;  // A的行数(M)和列数(K)

        int B_K, B_N;  // B的行数(K)和列数(N)

        

        if (!transA) {

            // A: [..., M, K]

            A_M = A_dims[A_rank - 2];

            A_K = A_dims[A_rank - 1];

        } else {

            // A转置: [..., K, M] -> 当作[..., M, K]使用

            A_M = A_dims[A_rank - 1];

            A_K = A_dims[A_rank - 2];

        }

        

        if (!transB) {

            // B: [..., K, N]

            B_K = B_dims[B_rank - 2];

            B_N = B_dims[B_rank - 1];

        } else {

            // B转置: [..., N, K] -> 当作[..., K, N]使用

            B_K = B_dims[B_rank - 1];

            B_N = B_dims[B_rank - 2];

        }

        

        // 检查K维度是否匹配

        if (A_K != B_K) {

            return std::nullopt;

        }

        

        // 准备输出形状

        Shape output_shape;

        

        // 处理广播维度

        if (A_rank == 2 && B_rank == 2) {

            // 简单2D情况

            output_shape = {A_M, B_N};

        } else {

            // 需要广播的情况

            

            // 获取batch维度

            size_t A_batch_dims = A_rank - 2;

            size_t B_batch_dims = B_rank - 2;

            

            // 确定最大batch维度数

            size_t max_batch_dims = std::max(A_batch_dims, B_batch_dims);

            

            // 从最前面的维度开始比较（广播是从前面的维度开始的）

            for (size_t i = 0; i < max_batch_dims; ++i) {

                // 计算在各自维度中的索引

                int64_t A_dim = (i < A_batch_dims) ? A_dims[i] : 1;

                int64_t B_dim = (i < B_batch_dims) ? B_dims[i] : 1;

                

                // 广播规则

                if (A_dim == B_dim) {

                    output_shape.push_back(A_dim);

                } else if (A_dim == 1) {

                    output_shape.push_back(B_dim);

                } else if (B_dim == 1) {

                    output_shape.push_back(A_dim);

                } else {

                    return std::nullopt;

                }

            }

            

            // 添加矩阵乘法的结果维度

            output_shape.push_back(A_M);

            output_shape.push_back(B_N);

        }

        

        return vector<Shape>{output_shape};        
    }

} // namespace infini