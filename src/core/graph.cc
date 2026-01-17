#include "core/graph.h"
#include <algorithm>
#include <iterator>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        
        // Use Kahn's algorithm for O(V+E) complexity
        std::unordered_map<OperatorObj *, int> inDegree;
        std::queue<Operator> zeroInDegree;
        std::vector<Operator> sorted;
        sorted.reserve(ops.size());
        inDegree.reserve(ops.size());
        
        // Calculate in-degree for each operator
        for (const auto &op : ops)
        {
            inDegree[op.get()] = 0;
        }
        for (const auto &op : ops)
        {
            for (const auto &succ : op->getSuccessors())
            {
                inDegree[succ.get()]++;
            }
        }
        
        // Initialize queue with operators having zero in-degree
        for (const auto &op : ops)
        {
            if (inDegree[op.get()] == 0)
            {
                zeroInDegree.push(op);
            }
        }
        
        // Process operators in topological order
        while (!zeroInDegree.empty())
        {
            auto current = zeroInDegree.front();
            zeroInDegree.pop();
            sorted.emplace_back(current);
            
            // Decrement in-degree for successors
            for (const auto &succ : current->getSuccessors())
            {
                if (--inDegree[succ.get()] == 0)
                {
                    zeroInDegree.push(succ);
                }
            }
        }
        
        // Check for cycles
        if (sorted.size() != ops.size())
        {
            return false;
        }
        
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
{
    // =================================== 作业 ===================================
    // TODO: 设计一个算法来实现指定的图优化规则
    // 图优化规则如下：
    // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
    // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
    // =================================== 作业 ===================================
    
    // Step 1: Remove redundant transpose operators
    bool modified = true;
    while (modified) {
        modified = false;
        for (auto it = ops.begin(); it != ops.end(); ) {
            auto op = *it;
            
            // Check if this is a transpose operator
            if (op->getOpType() != OpType::Transpose) {
                ++it;
                continue;
            }
            
            auto transposeOp = std::static_pointer_cast<TransposeObj>(op);
            auto permute = transposeOp->getPermute();
            
            // Check if the successor is also a transpose operator
            auto succs = op->getSuccessors();
            if (succs.size() != 1) {
                ++it;
                continue;
            }
            
            auto succOp = succs[0];
            if (succOp->getOpType() != OpType::Transpose) {
                ++it;
                continue;
            }
            
            auto succTranspose = std::static_pointer_cast<TransposeObj>(succOp);
            auto succPermute = succTranspose->getPermute();
            
            // Check if the two permutations are inverses of each other
            bool isInverse = (permute.size() == succPermute.size());
            for (size_t i = 0; i < permute.size() && isInverse; ++i) {
                if (permute[i] >= static_cast<int>(succPermute.size()) || 
                    succPermute[permute[i]] != static_cast<int>(i)) {
                    isInverse = false;
                    break;
                }
            }
            
            if (!isInverse) {
                ++it;
                continue;
            }
            
            // 获取输入输出tensor
            auto inputTensor = op->getInputs()[0];
            auto outputTensor = succOp->getOutputs()[0];
            auto intermediateTensor = op->getOutputs()[0];
            
            // 获取前驱和后继
            auto predecessor = inputTensor->getSource();
            
            // 更新连接关系
            if (predecessor) {
                // 前驱连接到后继的后继
                for (auto targetOp : succOp->getSuccessors()) {
                    predecessor->addSuccessors(targetOp);
                    targetOp->addPredecessors(predecessor);
                }
                predecessor->removeSuccessors(op);
            }
            
            // 更新tensor连接
            inputTensor->removeTarget(op);
            for (auto targetOp : outputTensor->getTargets()) {
                inputTensor->addTarget(targetOp);
                targetOp->replaceInput(outputTensor, inputTensor);
            }
            
            // 删除操作符
            auto nextIt = it;
            ++nextIt;
            
            // 确保不会重复删除
            if (std::find(ops.begin(), ops.end(), op) != ops.end()) {
                ops.erase(std::find(ops.begin(), ops.end(), op));
            }
            if (std::find(ops.begin(), ops.end(), succOp) != ops.end()) {
                ops.erase(std::find(ops.begin(), ops.end(), succOp));
            }
            
            // 删除中间tensor
            if (std::find(tensors.begin(), tensors.end(), intermediateTensor) != tensors.end()) {
                tensors.erase(std::find(tensors.begin(), tensors.end(), intermediateTensor));
            }
            if (std::find(tensors.begin(), tensors.end(), outputTensor) != tensors.end()) {
                tensors.erase(std::find(tensors.begin(), tensors.end(), outputTensor));
            }
            
            modified = true;
            it = ops.begin();  // 重新开始扫描
        }
    }
    
    // Step 2: Merge transpose into matmul operators
    modified = true;
    while (modified) {
        modified = false;
        for (auto it = ops.begin(); it != ops.end(); ) {
            auto op = *it;
            bool currentModified = false;
            
            if (op->getOpType() != OpType::MatMul) {
                ++it;
                continue;
            }
            
            auto matmulOp = std::static_pointer_cast<MatmulObj>(op);
            bool newTransA = matmulOp->getTransA();
            bool newTransB = matmulOp->getTransB();
            
            // Helper function to check and merge transpose
            auto tryMergeTranspose = [&](int inputIdx, bool& transFlag) -> bool {
                auto inputTensor = op->getInputs()[inputIdx];
                auto sourceOp = inputTensor->getSource();
                
                if (!sourceOp || sourceOp->getOpType() != OpType::Transpose) {
                    return false;
                }
                
                auto transposeOp = std::static_pointer_cast<TransposeObj>(sourceOp);
                auto permute = transposeOp->getPermute();
                auto shape = inputTensor->getDims();
                
                // Check if this transpose is only used by this matmul
                if (inputTensor->getTargets().size() > 1) {
                    return false;
                }
                
                // Check if transpose swaps only the last two dimensions and keeps others unchanged
                if (shape.size() < 2) {
                    return false;
                }
                
                size_t n = shape.size();
                bool isValidTranspose = true;
                
                // 检查是否是只交换最后两个维度的转置
                for (size_t i = 0; i < n; ++i) {
                    if (i < n - 2) {
                        // 前n-2个维度保持不变
                        if (permute[i] != static_cast<int>(i)) {
                            isValidTranspose = false;
                            break;
                        }
                    } else if (i == n - 2) {
                        // 倒数第二个维度交换到最后一个
                        if (permute[i] != static_cast<int>(n - 1)) {
                            isValidTranspose = false;
                            break;
                        }
                    } else { // i == n - 1
                        // 最后一个维度交换到倒数第二个
                        if (permute[i] != static_cast<int>(n - 2)) {
                            isValidTranspose = false;
                            break;
                        }
                    }
                }
                
                if (!isValidTranspose) {
                    return false;
                }
                
                // 可以合并
                transFlag = !transFlag;
                
                // 获取transpose的输入tensor
                auto bypassTensor = transposeOp->getInputs()[0];
                
                // 更新连接
                op->replaceInput(inputTensor, bypassTensor);
                bypassTensor->addTarget(op);
                inputTensor->removeTarget(op);
                
                // 删除transpose操作符
                auto transposeIt = std::find(ops.begin(), ops.end(), sourceOp);
                if (transposeIt != ops.end()) {
                    ops.erase(transposeIt);
                }
                
                // 删除中间tensor
                auto tensorIt = std::find(tensors.begin(), tensors.end(), inputTensor);
                if (tensorIt != tensors.end()) {
                    tensors.erase(tensorIt);
                }
                
                return true;
            };
            
            // Try merge transpose for input A
            if (tryMergeTranspose(0, newTransA)) {
                currentModified = true;
            }
            
            // Try merge transpose for input B
            if (tryMergeTranspose(1, newTransB)) {
                currentModified = true;
            }
            
            if (currentModified) {
                matmulOp->setTransA(newTransA);
                matmulOp->setTransB(newTransB);
                modified = true;
                it = ops.begin();  // 重新开始扫描
            } else {
                ++it;
            }
        }
    }
}
    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================
        

        std::vector<size_t> tensorOffsets; // 获取所有 tensor 的 offset

        for (auto &tensor : tensors) { // alloc() 必须在 getPtr 之前，所以用一个容器存下来
            size_t offset = allocator.alloc(tensor->getBytes());
            IT_ASSERT(offset != SIZE_MAX, "Memory allocation failed for tensor");
            tensorOffsets.emplace_back(offset);
        }

        auto ptr = allocator.getPtr();
        IT_ASSERT(ptr != nullptr, "Failed to get memory pointer from allocator");
        
        for(int i = 0; i < (int)tensors.size(); ++i) {
            auto blob = make_ref<BlobObj>(runtime, static_cast<char*>(ptr) + tensorOffsets[i]);
            tensors[i]->setDataBlob(blob);
        }
       
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini