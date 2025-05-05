from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
import os
import gevent.monkey
gevent.monkey.patch_all()
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.io.geventreactor import GeventConnection

app = Flask(__name__)

# 初始化 ScyllaDB 连接
def init_scylla_session():
    auth_provider = PlainTextAuthProvider(
        username=os.getenv("SCYLLA_USERNAME", "scylla"),
        password=os.getenv("SCYLLA_PASSWORD", "your-awesome-password")
    )
    contact_points = os.getenv("SCYLLA_CONTACT_POINTS", "127.0.0.1").split(",")
    cluster = Cluster(
        contact_points=contact_points,
        auth_provider=auth_provider,
        connection_class=GeventConnection  # 使用 GeventConnection
    )
    session = cluster.connect()
    return session

# 初始化全局 ScyllaDB 会话
session = init_scylla_session()

# 用户操作逻辑
def user_handler(todo, username, password, target):
    # 创建 keyspace 和 table
    session.execute("""
        CREATE KEYSPACE IF NOT EXISTS user WITH REPLICATION =
        {'class' : 'NetworkTopologyStrategy', 'replication_factor' : 1};
    """)
    session.execute("""
        CREATE TABLE IF NOT EXISTS user.info (
            name text,
            password text,
            PRIMARY KEY (name)
        );
    """)
    if todo == "create":
        # 检查用户是否存在
        rows = session.execute("SELECT name FROM user.info WHERE name = %s", (username,))
        if not rows.all():
            session.execute("INSERT INTO user.info (name, password) VALUES (%s, %s)", (username, password))
            target.append("success")
        else:
            target.append("exists")
    elif todo == "verify":
        # 验证用户名和密码
        rows = session.execute("SELECT password FROM user.info WHERE name = %s", (username,))
        for row in rows:
            if row.password == password:
                target.append("success")
                break
        else:
            target.append("failed")
    elif todo == "delete":
        # 验证用户名和密码
        rows = session.execute("SELECT password FROM user.info WHERE name = %s", (username,))
        for row in rows:
            if row.password == password:
                session.execute("DELETE FROM user.info WHERE name = %s", (username,))
                target.append("success")
                break
        else:
            target.append("failed")

@app.route('/user/create', methods=['POST'])
def create():
    try:
        # 从表单数据中提取用户名和密码
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            raise BadRequest("Missing username or password")

        # 处理用户创建逻辑
        result = []
        user_handler("create", username, password, result)
        return result[0]  # 直接返回字符串
    except Exception as e:
        return str(e), 400  # 返回错误信息的字符串

@app.route('/user/verify', methods=['POST'])
def verify():
    try:
        # 从表单数据中提取用户名和密码
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            raise BadRequest("Missing username or password")

        # 处理用户验证逻辑
        result = []
        user_handler("verify", username, password, result)
        return result[0]  # 直接返回字符串
    except Exception as e:
        return str(e), 400  # 返回错误信息的字符串

@app.route('/user/delete', methods=['POST'])
def delete():
    try:
        # 从表单数据中提取用户名和密码
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            raise BadRequest("Missing username or password")

        # 处理用户删除逻辑
        result = []
        user_handler("delete", username, password, result)
        return result[0]  # 直接返回字符串
    except Exception as e:
        return str(e), 400  # 返回错误信息的字符串

# 新增图片预测路由
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从表单数据中提取用户名、密码和图片
        username = request.form.get('username')
        password = request.form.get('password')
        image = request.files.get('image')
        if not username or not password or not image:
            raise BadRequest("Missing username, password or image")

        # 保存上传的图片
        filename = secure_filename(image.filename)
        image_path = os.path.join("uploads", filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path)

        # 验证用户名和密码
        result = []
        user_handler("verify", username, password, result)
        if result[0] != "success":
            return "User verification failed", 401

        # 调用图片预测函数（这里假设有一个 predict 函数）
        from fine_tuned_resnet50 import predict  # 假设这是你的预测模块
        prediction_result = predict(image_path)
        # return jsonify({"prediction": prediction_result})
        return prediction_result  # 直接返回字符串
    except Exception as e:
        return str(e), 400  # 返回错误信息的字符串

if __name__ == '__main__':
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('0.0.0.0', 5800), app)
    print("Server running on http://0.0.0.0:5800")
    http_server.serve_forever()