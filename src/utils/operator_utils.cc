#include "utils/operator_utils.h"
#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {

    // =================================== 作业 ===================================
    // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
    // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    // =================================== 作业 ===================================
    // 获取A和B的维度数

    int64_t dimsA = static_cast<int64_t>(A.size());
    int64_t dimsB = static_cast<int64_t>(B.size());
    // 计算广播后的维度数（取两者中的最大值）
    int64_t maxDims = std::max(dimsA, dimsB);
    // 创建结果形状
    Shape result(maxDims);
    // 从最右侧维度开始比较
    for (int64_t i = 0; i < maxDims; i++) {
        // 计算A和B在当前维度上的索引（从右侧开始）
        int64_t idxA = dimsA - 1 - i;
        int64_t idxB = dimsB - 1 - i;
        // 获取A和B在当前维度上的大小（如果维度不存在则为1）
        int64_t dimA = (idxA >= 0) ? A[idxA] : 1;
        int64_t dimB = (idxB >= 0) ? B[idxB] : 1;
        // 计算结果维度
        if (dimA == dimB) {
            // 情况1：维度大小相等
            result[maxDims - 1 - i] = dimA;
        } else if (dimA == 1) {
            // 情况2：A的维度为1，使用B的维度
            result[maxDims - 1 - i] = dimB;
        } else if (dimB == 1) {
            // 情况3：B的维度为1，使用A的维度
            result[maxDims - 1 - i] = dimA;
        } else {
            // 情况4：维度大小不相等且都不为1，无法广播
            throw std::invalid_argument(
                "Incompatible shapes for broadcasting: dimension " + 
                std::to_string(i) + " is " + std::to_string(dimA) + 
                " and " + std::to_string(dimB) + ", but both are > 1"
            );
        }
    }
    return result;    
}

int get_real_axis(const int &axis, const int &rank) {
    IT_ASSERT(rank >= 1);
    IT_ASSERT(axis >= -rank && axis <= (rank - 1));
    int newAxis;
    if (axis < 0) {
        newAxis = rank + axis;
    } else {
        newAxis = axis;
    }
    return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
    Shape ans(shape.size());
    auto i = ans.rbegin();
    auto j = shape.rbegin(), ej = shape.rend();
    while (j != ej) {
        auto div = std::div(inputN, *j++);
        *i++ = div.rem;
        inputN = div.quot;
    }
    return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape,
                      const Shape &stride) {
    size_t ans = 0;
    Shape index(shapeIndex.size());
    IT_ASSERT(shapeIndex.size() == shape.size());
    IT_ASSERT(shape.size() == stride.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        index[i] = shapeIndex[i] % shape[i];
        ans += index[i] * stride[i];
    }
    return ans;
}

std::string device_to_str(Device device) {
    std::string deviceStr;
    switch (device) {
    case Device::CPU:
        return "CPU";
    default:
        IT_TODO_HALT();
    }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
    std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
    std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
    return deviceStr + ", " + opStr;
}

} // namespace infini
