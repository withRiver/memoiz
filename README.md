 # `memoiz` - 自动记忆化搜索宏

 `memoiz` 是一个 Rust 属性宏，用于将函数转换为**记忆化搜索**（memoization），通过缓存函数调用结果，避免重复计算，显著优化递归函数性能。

 该宏适用于需要减少重复计算的场景（如斐波那契数列、动态规划问题），且支持多线程安全（内部使用 `Mutex` 保护缓存）。

 ## 特性

 - **自动缓存**：首次调用时计算并缓存结果，后续调用直接返回缓存值。
 - **线程安全**：基于 `Mutex` 实现，支持多线程并发访问。
 - **惰性初始化**：缓存仅在第一次调用时初始化，节省资源。
 - **参数限制**：函数参数必须实现 `Clone + PartialEq + Eq + Hash`（默认要求）。
 
 ## 更新
 
 v0.2: 参数中的不可变引用参数不会被缓存，从而支持了被标注函数参数中有不可变引用（如Vec, Slice等不可变引用）的使用场景。
 
 ## 示例

### Example 1
 ```rust
 use memoiz::memo;

 #[memo]
 fn fib(n: u32) -> u32 {
    if n <= 1 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
 }

fn main() {
    assert_eq!(fib(10), 55);
}
 ```
### Example 2
```rust
use std;
use memoiz::memo;

#[memo]
fn f(nums: &Vec<i32>, n: usize) -> i32{
    if n == 0 {
        return 0;
    }

    if n == 1 {
        return nums[0];
    }

    return std::cmp::max(f(nums, n - 2) + nums[n - 1], f(nums, n - 1));
}

fn main() {
    let nums = vec![2, 7, 9, 3, 1];
    assert_eq!(f(&nums, nums.len()), 12)
}
```

 ## 注意事项

 1. **参数类型要求**：
    - 函数参数必须支持 `Clone + PartialEq + Eq + Hash`，否则无法作为缓`存键。
    - 如果参数包含不可克隆/不可比较的类型（如 `Vec<u8>`），需手动包装或改用其他缓存策略。（若在每次递归调用中该参数值不变，那么可以传入不可变引用）

 2. **线程安全**：
    - 缓存使用 `Mutex` 保护，确保多线程环境下安全访问。
    - 如果函数本身是线程不安全的（如修改全局状态），需自行处理同步逻辑。

 3. **性能权衡**：
    - 缓存会占用内存，适合调用次数多、计算开销大的函数。
    - 小参数范围或短生命周期的函数可能不适合使用。

 4. **递归限制**：
    - 宏会自动展开递归调用，但需确保递归终止条件正确。

 ## 属性说明

 - `#[memo]`：直接标注在函数上，自动生成缓存逻辑。
 

 ## 使用方法

 在 `Cargo.toml` 中添加依赖：

 ```toml
 [dependencies]
 memoiz = "0.1.0"
 ```


