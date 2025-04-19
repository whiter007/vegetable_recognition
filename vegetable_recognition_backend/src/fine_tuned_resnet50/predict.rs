use candle_core::{
    D, // 为Tensor类型提供索引操作，用于获取张量的某个维度
    DType,
    Device,
    Error,
    IndexOp,
    Tensor,
}; // 维度
use candle_nn::{
    Module, // 模型trait, 用于实现前向传播方法
    VarBuilder,
}; // 模型参数的反序列化
// use clap::{Parser, ValueEnum};
use image::ImageReader;
use super::dataset::CLASSES; // 本地的模型数据集类别
use super::model::resnet50; // 本地的微调模型

// use once_cell::sync::Lazy; // 用于实现全局的模型参数
// static MODEL: Lazy<candle_nn::Func> = Lazy::new(|| {
//     let device = Device::new_cuda(0).expect("msg: failed to get device");
//     let binding: std::path::PathBuf = std::env::current_exe()
//         .expect("msg: failed to get current exe")
//         .parent()
//         .unwrap()
//         .join("model.safetensors");
//     let model_file: &str = binding.to_str().unwrap();

//     let vb = unsafe {
//         VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)
//             .expect("msg: failed to load model parameters")
//     };
//     println!("model parameters loaded");
//     let class_count = 36;
//     resnet50(class_count, vb).expect("msg: failed to build model")
// });

pub const IMAGENET_MEAN: [f32; 3] = [0.485f32, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229f32, 0.224, 0.225];
pub fn predict(image_path: &str) -> anyhow::Result<String> {
    // ----- 预处理 -----
    let device = Device::new_cuda(0)?;
    // ----- 图像加载与图像处理 -----
    let image = image_path;
    // println!("正在预处理图片");
    let processed_image: Tensor = {
        let img = ImageReader::open(image)?
            .decode()
            .map_err(Error::wrap)?
            .resize_to_fill(
                224 as u32,
                224 as u32,
                image::imageops::FilterType::Triangle,
            );
        let img = img.to_rgb8();
        let data = img.into_raw();
        let data = Tensor::from_vec(data, (224, 224, 3), &device)?.permute((2, 0, 1))?;
        let mean = Tensor::new(&IMAGENET_MEAN, &device)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&IMAGENET_STD, &device)?.reshape((3, 1, 1))?;
        (data.to_dtype(DType::F32)? / 255.)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?
    };
    // println!("图片处理完成");

    // ----- 反序列化模型参数 与 模型构建 -----
    // let model = MODEL.to_owned(); // 使用全局的模型参数
    let binding: std::path::PathBuf = std::env::current_exe()?
        .parent()
        .unwrap()
        .join("model.safetensors");
    let model_file: &str = binding.to_str().unwrap();

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
    // println!("模型参数已加载");
    let class_count = 36;
    let model = resnet50(class_count, vb)?;
    // println!("模型已构建");
    // ----- 模型推理 -----
    let logits = model.forward(&processed_image.unsqueeze(0)?)?;
    // println!("{:#?}", logits); // 打印输出层张量
    // ----- 处理Tensor输出对应分类 -----
    let predicts = candle_nn::ops::softmax(&logits, D::Minus1)?
        .i(0)?
        .to_vec1::<f32>()?; //对输出层张量使用softmax激活，存入一个一维向量
    // println!("{:#?}", prs); // 打印概率分布

    let mut predicts: Vec<(usize, &f32)> = predicts.iter().enumerate().collect::<Vec<_>>(); //为每个类别的概率分布添加索引
    // println!("{:#?}", prs); // 打印带有索引的概率分布元组

    predicts.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1)); //按照概率大小进行排序
    // println!("{:#?}", prs); // 打印排序后的概率分布元组

    let predict_result = predicts
        .iter()
        .take(5)
        .map(|&(category_idx, predict_rate)| {
            format!("{}: {:.2}%", CLASSES[category_idx], 100. * predict_rate)
        }) // 保留两位小数
        .collect::<Vec<_>>()
        .join("\n");
    Ok(predict_result)
}
