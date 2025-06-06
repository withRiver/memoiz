//! # `memoiz` - 自动记忆化搜索宏
//!
//! `memoiz` 是一个 Rust 属性宏，用于将函数转换为**记忆化搜索**（memoization），通过缓存函数调用结果，避免重复计算，显著优化递归函数性能。
//!
//! 该宏适用于需要减少重复计算的场景（如斐波那契数列、动态规划问题），且支持多线程安全（内部使用 `Mutex` 保护缓存）。
//!
//! ## 特性
//!
//! - **自动缓存**：首次调用时计算并缓存结果，后续调用直接返回缓存值。
//! - **线程安全**：基于 `Mutex` 实现，支持多线程并发访问。
//! - **惰性初始化**：缓存仅在第一次调用时初始化，节省资源。
//! - **参数限制**：函数参数必须实现 `Clone + PartialEq + Eq + Hash`（默认要求）。
//!
//! ## 示例
//!
//! ```rust
//! use memoiz::memo;
//!
//! #[memo]
//! fn fib(n: u32) -> u32 {
//!     if n <= 1 {
//!         n
//!     } else {
//!         fib(n - 1) + fib(n - 2)
//!     }
//! }
//!
//! assert_eq!(fib(10), 55);
//!    
//! ```
//!
//! ## 注意事项
//!
//! 1. **参数类型要求**：
//!    - 函数参数必须支持 `Clone + PartialEq + Eq + Hash`，否则无法作为缓`存键。
//!    - 如果参数包含不可克隆/不可比较的类型（如 `Vec<u8>`），需手动包装或改用其他缓存策略。
//!
//! 2. **线程安全**：
//!    - 缓存使用 `Mutex` 保护，确保多线程环境下安全访问。
//!    - 如果函数本身是线程不安全的（如修改全局状态），需自行处理同步逻辑。
//!
//! 3. **性能权衡**：
//!    - 缓存会占用内存，适合调用次数多、计算开销大的函数。
//!    - 小参数范围或短生命周期的函数可能不适合使用。
//!
//! 4. **递归限制**：
//!    - 宏会自动展开递归调用，但需确保递归终止条件正确。
//!
//! ## 属性说明
//!
//! - `#[memo]`：直接标注在函数上，自动生成缓存逻辑。
//! 
//!
//! ## 使用方法
//!
//! 在 `Cargo.toml` 中添加依赖：
//!
//! ```toml
//! [dependencies]
//! memoiz = "0.1.0"
//! ```
//!
//!
//!

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse_macro_input, spanned::Spanned, Error, FnArg, Ident, ItemFn, Pat, PatIdent, ReturnType, 
    parse::{Parse, ParseStream}, punctuated::Punctuated, Token
};
use std::collections::HashSet;

struct KeyArgs {
    args: Vec<Ident>, 
}

impl Parse for KeyArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let args = Punctuated::<Ident, Token![,]>::parse_terminated(input)?;
        Ok(Self {
            args: args.into_iter().collect(),
        })
    }
}


#[proc_macro_attribute]
pub fn memo(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let key_arg_names = parse_macro_input!(attr as KeyArgs).args;

    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.vis;
    let fn_block = &input_fn.block;
    let fn_inputs = &input_fn.sig.inputs;
    let fn_output = &input_fn.sig.output;
    let fn_sig = &input_fn.sig;

    // 无缓存版本的函数名: func_no_cache
    let no_cache_name = Ident::new(&format!("{}_no_cache", fn_name), fn_name.span());

    // 哈希表的名字
    let cache_name = Ident::new(&fn_name.to_string().to_uppercase(), fn_name.span());

    let _global_cache_name = Ident::new(&format!("global_cache_{}", fn_name), fn_name.span());

    // 提取参数和类型
    let (args, param_types): (Vec<_>, Vec<_>) = input_fn
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Typed(pat_type) => {
                // 提取参数标识符
                let ident = match &*pat_type.pat {
                    Pat::Ident(PatIdent { ident, .. }) => ident.clone(),
                    _ => {
                        return Err(Error::new(
                            pat_type.span(),
                            "only simple identifiers are supported",
                        ))
                    }
                };
                // 提取参数类型
                let ty = &*pat_type.ty;
                Ok((ident, ty))
            }
            _ => Err(Error::new(arg.span(), "self parameters are not supported")),
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .into_iter()
        .unzip();
    
    if args.is_empty() { return quote! {#fn_vis #fn_sig #fn_block}.into() ; }

    let key_arg_names_set: HashSet<String> = key_arg_names.iter().map(|id| id.to_string()).collect();
    let use_all_args = key_arg_names_set.is_empty();

    let (key_args, key_types): (Vec<_>, Vec<_>) = args.iter()
        .zip(param_types.into_iter())
        .filter(|(arg, _)| use_all_args || key_arg_names_set.contains(&arg.to_string()))
        .map(|(arg, ty)| (arg.clone(), ty.clone()))
        .unzip();   

    // 检查无效的参数名
    if !use_all_args {
        let arg_names: HashSet<String> = args.iter().map(|a| a.to_string()).collect();
        for name in &key_arg_names {
            if !arg_names.contains(&name.to_string()) {
                return Error::new(name.span(), format!("'{}' is not a function parameter", name))
                    .to_compile_error()
                    .into();
            }
        }
    }


    let key_type = if key_types.len() == 1 {
        quote! { #(#key_types)* }
    } else {
        quote! { (#(#key_types),*) }
    };

    //key = (arg1.clone(), arg2.clone())
    let key_exprs = key_args.iter().map(|arg| quote! { #arg.clone() });
    let key_tuple = quote! { (#(#key_exprs),*) };

    // (arg1, arg2, ...)
    let call_args = args.iter().map(|arg| quote! { #arg });


    let return_type = match fn_output {
        ReturnType::Default => quote! { () },
        ReturnType::Type(_, ty) => quote! { #ty },
    };



    let create_cache = quote! {
        static #cache_name: ::std::sync::LazyLock<::std::sync::Mutex<::std::collections::HashMap<#key_type, #return_type>>> = ::std::sync::LazyLock::new(|| {
        ::std::sync::Mutex::new(::std::collections::HashMap::new())
});
    };

    let no_cache_fn = quote! {
        #fn_vis fn #no_cache_name(#fn_inputs) #fn_output #fn_block
    };

    let cached_fn = quote! {
        #fn_vis fn #fn_name(#fn_inputs) #fn_output {
            let key = #key_tuple;
            // 检查缓存
            {   
                let cache = #cache_name.lock().unwrap();
                if let Some(result) = cache.get(&key) {
                    return result.clone();
                }
            }
            // 计算并缓存结果
            let result = #no_cache_name(#(#call_args),*);
            let mut cache = #cache_name.lock().unwrap();
            cache.insert(key, result.clone());
            result
        }
    };

    let expanded = quote! {
        #create_cache
        #no_cache_fn
        #cached_fn
    };

    expanded.into()
}