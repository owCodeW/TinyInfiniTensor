#include "operators/transpose.h"

namespace infini
{
    TransposeObj::TransposeObj(GraphObj *graph, Tensor input, Tensor output,
                               vector<int> permute)
        : OperatorObj(OpType::Transpose, {input}, {output})
    {
        auto rank = input->getRank();
        if (permute.empty())
        {
            for (size_t i = 0; i < rank; ++i)
            {
                transposePermute[i] = i;
            }
        }
        else
        {
            IT_ASSERT(rank == permute.size());
            transposePermute = std::move(permute);
        }
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> TransposeObj::inferShape(const TensorVec &inputs)
    {
       

        // =================================== 作业 ===================================
        // TODO：修改 output_dim，返回正确的 transpose 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Transpose.html#transpose-21
        // =================================== 作业 ===================================
        // 1. 检查输入参数
        if (inputs.size() != 1) {
             // 通常应该记录日志或抛出异常
            return std::nullopt;
        }
        const auto A = inputs[0];
        auto input_dim = A->getDims();
        int rank = A->getRank();
        // 2. 验证 permute 参数的有效性
        if (transposePermute.size() != static_cast<size_t>(rank)) {
            // 错误：permute 数组长度必须等于 rank
            return std::nullopt;
        }
        // 3. 创建输出形状并计算
        vector<int> output_dim(rank);
        for (int i = 0; i < rank; ++i) {
            int src_idx = transposePermute[i];
            // 检查索引是否越界
            if (src_idx < 0 || src_idx >= rank) {
                return std::nullopt;  // 索引越界
            }
            output_dim[i] = input_dim[src_idx];
        }
        return vector<Shape>{output_dim};  // 注意：这里返回的是 vector<Shape>，需要嵌套一层        
    }

    std::string TransposeObj::toString() const
    {
        std::ostringstream os;
        os << type.toString() << "[" << getGuid() << "]";
        os << "(";
        os << vecToString(inputs[0]->getDims()) << ",";
        os << "input=" << inputs[0]->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }
}; // namespace infini
