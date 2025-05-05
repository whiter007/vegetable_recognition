from locust import HttpUser, task, between
import os

class PredictUser(HttpUser):
    # 设置用户任务之间的等待时间
    wait_time = between(1, 3)

    @task
    def test_predict_api(self):
        # 定义表单数据
        files = {'image': open('Image_1.jpg', 'rb')}
        data = {
            'username': 'test_user',
            'password': 'P@ssw0rd123'
        }
        try:
            # 发送 POST 请求到预测接口
            response = self.client.post("/predict", data=data, files=files)
            if response.status_code == 200:
                print("请求成功")
            else:
                print(f"请求失败，状态码: {response.status_code}")
        except Exception as e:
            print(f"请求发生错误: {e}")