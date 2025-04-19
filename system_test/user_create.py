from locust import HttpUser, TaskSet, task, between
import random
import string

class UserBehavior(TaskSet):
    @task
    def create_user(self):
        # 生成随机用户名
        username = "test_user_" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        password = "P@ssw0rd123"

        # 发送POST请求
        with self.client.post(
            "/user/create",
            data={"username": username, "password": password},
            catch_response=True
        ) as response:
            # 检查响应内容
            if response.text == "success":
                response.success()
            else:
                response.failure(f"Unexpected response: {response.text}")

class WebsiteUser(HttpUser):
    host = "http://localhost:5800"  # 直接在代码中定义host
    tasks = [UserBehavior]
    wait_time = between(1, 5)  # 每个用户请求之间的等待时间在1到5秒之间