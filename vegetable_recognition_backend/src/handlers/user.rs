// use crate::ScyllaDB;
// use salvo::basic_auth::{BasicAuth, BasicAuthValidator};
use salvo::prelude::*;

use crate::scylla_db::user_handler;
pub fn user_route() -> Router {
    Router::with_path("user")
        // .push(Router::with_path("new").post(handlers::user::new))
        .push(Router::with_path("create").post(create))
        .push(Router::with_path("verify").post(verify))
        .push(Router::with_path("delete").post(delete))
}

#[allow(unused)]
#[handler]
async fn create(req: &mut Request, res: &mut Response) {
    // 从请求中提取 multipart 表单数据
    let form_data = req.form_data().await.unwrap();
    let username = form_data.fields.get("username").unwrap();
    let password = form_data.fields.get("password").unwrap();

    // 验证用户名和密码
    let mut result = String::from("正在创建用户名和密码");
    user_handler("create", username, password, &mut result).await;

    res.render(&result);
}

#[allow(unused)]
#[handler]
async fn verify(req: &mut Request, res: &mut Response) {
    // 从请求中提取 multipart 表单数据
    let form_data = req.form_data().await.unwrap();
    let username = form_data.fields.get("username").unwrap();
    let password = form_data.fields.get("password").unwrap();

    // 验证用户名和密码
    let mut result = String::from("正在验证用户名和密码");
    user_handler("verify", username, password, &mut result).await;

    res.render(&result);
}

#[allow(unused)]
#[handler]
async fn delete(req: &mut Request, res: &mut Response) {
    // 从请求中提取 multipart 表单数据
    let form_data = req.form_data().await.unwrap();
    let username = form_data.fields.get("username").unwrap();
    let password = form_data.fields.get("password").unwrap();

    // 验证用户名和密码
    let mut result = String::from("正在删除用户名和密码");
    user_handler("delete", username, password, &mut result).await;

    res.render(&result);
}
