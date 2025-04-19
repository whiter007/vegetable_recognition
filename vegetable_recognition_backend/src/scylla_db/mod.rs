use futures::TryStreamExt;
use scylla::client::session::Session;
use scylla::client::session_builder::SessionBuilder;
use std::error::Error;
use tokio::sync::OnceCell;
// use uuid::Uuid;

static SESSION: OnceCell<Session> = OnceCell::const_new();
// #[tokio::main]
pub async fn user_handler(
    // session: &Session,
    todo: &str,
    user: &str,
    password: &str,
    target: &mut String,
) -> Result<(), Box<dyn Error>> {
    // target.push_str("\n进入异步函数成功");
    // let uri = std::env::var("SCYLLA_URI").unwrap_or_else(|_| "127.0.0.1:9042".to_string());
    // let session: Session = SessionBuilder::new()
    //     .known_node(uri)
    //     .build()
    //     .await
    //     .expect("msg: failed to connect to ScyllaDB");
    let session = SESSION
        .get_or_init(|| async {
            let uri = std::env::var("SCYLLA_URI").unwrap_or_else(|_| "127.0.0.1:9042".to_string());
            SessionBuilder::new()
                .known_node(uri)
                .build()
                .await
                .expect("msg: failed to connect to ScyllaDB")
        })
        .await; //使用全局的session

    // target.push_str("\n服务器连接成功");
    session
        .query_unpaged(
            "CREATE KEYSPACE IF NOT EXISTS user WITH REPLICATION = \
{'class' : 'NetworkTopologyStrategy', 'replication_factor' : 1}",
            &[],
        )
        .await?; //检查并创建user这个keyspace

    // target.push_str("\n操作执行成功");
    session
        .query_unpaged(
            "CREATE TABLE IF NOT EXISTS user.info (name text, password text, PRIMARY KEY (name))",
            &[],
        )
        .await?; //检查并创建user这个keyspace下的info这个表

    if todo == "create" {
        let exist = session
            .query_iter(
                "SELECT name, password FROM user.info WHERE name = ?",
                (user,),
            )
            .await?
            .rows_stream::<(String, String)>()?
            .try_next()
            .await?
            .is_some(); //检查user这个keyspace下的info这个表里是否存在该用户的数据
        if !exist {
            session
                .query_unpaged(
                    "INSERT INTO user.info (name, password) VALUES(?, ?)",
                    (user, password),
                )
                .await?;
            target.clear();
            target.push_str("success"); //如果不存在则插入该用户的信息
        } else {
            target.clear();
            target.push_str("exists"); //如果存在则返回exists
        }
    } else if todo == "verify" {
        session
            .query_iter(
                "SELECT name, password FROM user.info WHERE name = ?",
                (user,),
            )
            .await?
            .rows_stream::<(String, String)>()?
            .try_next()
            .await?
            .map(|(name_data, password_data)| {
                if name_data == user && password_data == password {
                    target.clear();
                    target.push_str("success"); //如果等同则返回sucess
                } else {
                    target.clear();
                    target.push_str("failed"); //如果不同则返回failed
                }
            });
    } else if todo == "delete" {
        let data = session
            .query_iter(
                "SELECT name, password FROM user.info WHERE name = ?",
                (user,),
            )
            .await?
            .rows_stream::<(String, String)>()?
            .try_next()
            .await?; // 获取该用户的数据
        let verified = match data {
            Some(_) => {
                let data = data.unwrap();
                if &data.0 == user && &data.1 == password {
                    true
                } else {
                    false
                }
            }
            None => false,
        }; // 验证用户名和密码

        if verified {
            session
                .query_unpaged("DELETE FROM user.info WHERE name = ?", (user,))
                .await?;
            target.clear();
            target.push_str("success"); //如果验证成功则删除用户
        } else {
            target.clear();
            target.push_str("failed"); //如果验证失败则返回failed
        }
    }

    Ok(())
}
