use crate::fine_tuned_resnet50::predict::predict;
use crate::scylla_db::user_handler;
use salvo::prelude::*;

#[allow(unused)]
#[handler]
pub async fn predict_handler(req: &mut Request, res: &mut Response) {
    // 格式要求为multipart/form-data混合传输文本和文件的标准格式
    // println!("收到图片处理请求");

    // 从请求中提取 multipart 表单数据
    let form_data = req.form_data().await.unwrap();
    let username = form_data.fields.get("username").unwrap();
    let password = form_data.fields.get("password").unwrap();
    let image = form_data
        .files
        .get("image")
        .unwrap()
        .path()
        .to_str()
        .unwrap();
    println!("图片路径已加载: {}", image);
    // 验证用户名和密码
    let mut result = String::from("正在验证用户名和密码");
    user_handler("verify", username, password, &mut result).await;
    println!("用户已验证");
    // 预测图片
    if result == "success" {
        result = predict(image).unwrap();
    }
    println!("图片已预测");
    res.render(&result);
}
