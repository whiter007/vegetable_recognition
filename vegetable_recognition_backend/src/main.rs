mod fine_tuned_resnet50;
mod handlers;
mod scylla_db;

use salvo::prelude::*;

#[tokio::main]
async fn main() {
    let user_router = handlers::user::user_route();
    let predict_router = Router::with_path("predict").post(handlers::predict::predict_handler);
    let router = Router::new().push(user_router).push(predict_router);

    let acceptor = TcpListener::new("0.0.0.0:5800").bind().await;

    Server::new(acceptor).serve(router).await;
}
