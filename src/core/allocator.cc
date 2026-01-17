#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        // 1. 在 freeBlocks 中查找合适块--首次适配算法
        for (auto it = freeBlocks.begin(); it != freeBlocks.end(); ++it) {
            if (it->second >= size) { // 找到大小足够的内存块
                size_t start_addr = it->first;
                size_t block_size = it->second;  // 空闲块的原始大小
                // 从空闲块列表中移除这个块
                freeBlocks.erase(it);
                // 检查是否需要分割
                if (block_size > size)
                {
                    // 分割：剩余部分作为新的空闲块
                    size_t remaining_addr = start_addr + size;
                    size_t remaining_size = block_size - size;
                    freeBlocks[remaining_addr] = remaining_size;
                }
                // 如果 block_size == size，就不需要创建新的空闲块
                // 更新已使用内存
                used += size;
                // 更新 peak（检查当前分配是否达到了新的峰值）
                size_t end_addr = start_addr + size;
                if (end_addr > peak) {
                    peak = end_addr;
                }
                return start_addr;
            }
        }
        // 2. freeBlocks 中没有合适块，需要从末尾分配
        // 末尾地址就是当前的 peak
        auto it = freeBlocks.begin();
        size_t start_addr;
        if(it->first + it-> second != this->peak) {
            start_addr = this->peak;  // 从当前 peak 开始分配
            size_t end_addr = start_addr + size;
            // 更新 peak
            this->peak = end_addr;
            used += size;
        } else {  // 末尾拓展
            freeBlocks.erase(it);
            start_addr = it->first;
            this->peak = it->first + size;
            used += size;

        }
        return start_addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        // 将释放的内存块加入freeBlocks
        this->freeBlocks[addr] = size;
        this->used -= size;
        // 合并相邻的空闲块
        auto it = this->freeBlocks.find(addr);
        // 检查是否能与前一个块合并
        if (it != this->freeBlocks.begin()) {
            auto prev = std::prev(it);
            if (prev->first + prev->second == it->first) {  // first: 起始地址 second: 大小
                // 可以合并
                prev->second += it->second; // 合并时更新大小
                this->freeBlocks.erase(it);
                it = prev;  // 更新当前迭代器指向合并后的块
            } else {
                // 不能合并，恢复it指向原始块
                it = freeBlocks.find(addr);
            }
        }

        // 检查是否能与后一个块合并
        if (it != freeBlocks.end()) {
            auto next = std::next(it);
            if (next != freeBlocks.end() && it->first + it->second == next->first) {
                // 可以合并
                it->second += next->second;
                freeBlocks.erase(next);
            }
        }
        // 之所以可以这么合并，因为map中key是有序分布的
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
