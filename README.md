# 快速使用
创建一个ScyllaDB数据库容器，
点击start_database.bat启动数据库，
点击vegetable_recognition_backend.exe启动后端，确保model.safetensors在可执行文件同级目录下，

将app-debug.apk安装到Android设备上,
打开软件输入可访问的后端地址即可开始使用

# 源码说明
已去除训练数据集，如需训练可按照保留的验证集格式添加数据
rust后端调试需把model.safetensors放在CARGO_TARGET_DIR\debug\model.safetensors
flutter前端只能在debug模式下运行


# 1 启动服务器 startup server
## 1.1 数据库 database
### 1.1.1 第一次启动 first time
```cmd
docker pull scylladb/scylla
docker run --name scylla -p 9042:9042 -d scylladb/scylla
```
### 1.1.2 后续启动 subsequent time
```cmd
docker start scylla
```
## 1.2 启动后端 startup backend
将`model.safetensors`放在可执行文件`vegetable_recognition_backend.exe`同级目录下，然后运行`vegetable_recognition_backend.exe`

put `model.safetensors` in the same directory as the executable file `vegetable_recognition_backend.exe`, and then run `vegetable_recognition_backend.exe`

# 2 启动前端 startup frontend

## 2.1 编译Android应用 compile Android app

### 2.1.1 安装依赖 install dependencies
```cmd
flutter pub get
```
### 2.1.2 编译 compile
```cmd
flutter build apk
```
## 2.2 安装Android应用 install Android app
将`build/app/outputs/apk/release/app-debug.apk`或`build/app/outputs/flutter-apk/app-debug.apk`安装到Android设备上

put `build/app/outputs/apk/release/app-release.apk` or `build/app/outputs/flutter-apk/app-debug.apk` on Android device