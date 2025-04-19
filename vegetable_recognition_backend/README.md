# 服务器地址

此后端程序
`http://localhost:5800`

数据库需开放端口
使用`docker pull scylladb/scylla`拉取数据库镜像
使用`docker run --name scylla -p 9042:9042 -d scylladb/scylla`启动数据库容器
.

# 用户创建接口

- 请求方式: POST

- 接口路径: `/user/create`

- 请求参数（表单格式）:

`username`: 字符串类型，必填，用户名称
`password`: 字符串类型，必填，登录密码

- 请求示例:
提交包含 `username=test_user` 和 `password=P@ssw0rd123` 的表单数据
```
curl -X POST "http://localhost:5800/user/create" -d "username=test_user" -d "password=P@ssw0rd123"
```

- 响应示例:
成功返回字符串 `success` 失败返回字符串 `exists`


# 用户验证接口

- 请求方式: POST

- 接口路径: `/user/verify`

- 请求参数（表单格式）:

`username`: 字符串类型，必填

`password`: 字符串类型，必填

- 请求示例:
提交包含 `username=test_user` 和 `password=P@ssw0rd123` 的表单数据
```
curl -X POST "http://localhost:5800/user/verify" -d "username=test_user" -d "password=P@ssw0rd123"
```

- 响应示例:
成功返回字符串 `success` 失败返回字符串 `failed`


# 用户删除接口

- 请求方式: POST

- 接口路径: /user/delete

- 请求参数（表单格式）:

`username`: 字符串类型，必填

`password`: 字符串类型，必填

- 请求示例:
提交包含 `username=test_user` 和 `password=P@ssw0rd123` 的表单数据
```
curl -X POST "http://localhost:5800/user/delete" -d "username=test_user" -d "password=P@ssw0rd123"
```

- 响应示例:
成功返回字符串 `success` 失败返回字符串 `failed`


# 预测功能接口

- 请求方式: POST

- 接口路径: /predict

- 请求参数（multipart/form-data）:

`username`: 字符串类型，必填

`password`: 字符串类型，必填

`image`: 文件类型，必填，支持 JPG/PNG 格式


- 请求示例:
提交包含 `username=test_user` 和 `password=P@ssw0rd123` 和图片文件的混合表单数据，@ 符号告诉 curl 从本地路径读取文件
```
curl -X POST "http://localhost:5800/predict" -F "username=test_user" -F "password=P@ssw0rd123" -F "image=@Image_1.jpg"
```


- 响应示例:
成功返回 五个预测结果 字符串
如
```
cabbage卷心菜: 87.51%
potato土豆: 0.62%
bell pepper柿子椒: 0.60%
mango芒果: 0.56%
banana香蕉: 0.56%
```
失败返回字符串 `failed`
