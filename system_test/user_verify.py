from locust import HttpUser, TaskSet, task, between

class UserVerificationBehavior(TaskSet):
    @task
    def verify_user(self):
        # 设置用户名和密码
        username = "whiter"
        password = "password"

        # 发送POST请求
        with self.client.post(
            "/user/verify",
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
    tasks = [UserVerificationBehavior]
    wait_time = between(1, 5)  # 每个用户请求之间的等待时间在1到5秒之间