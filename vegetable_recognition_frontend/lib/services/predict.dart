import 'dart:io';
import 'package:dio/dio.dart';
// import 'package:http_parser/http_parser.dart';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
class PredictService {
  final Dio _dio = Dio()
    ..interceptors.add(InterceptorsWrapper(
      onRequest: (options, handler) {
        debugPrint('请求URI: ${options.uri}');
        debugPrint('请求头: ${options.headers}');
        if (options.data is FormData) {
          debugPrint('表单字段: ${(options.data as FormData).fields}');
          debugPrint('文件字段: ${(options.data as FormData).files}');
        }
        return handler.next(options);
      },
    ));

  Future<String> imagePredict(String imagePath) async {
    final prefs = await SharedPreferences.getInstance();

    try {
      final username = prefs.getString('username');
      final password = prefs.getString('password');
      final serverAddress = prefs.getString('server_address');

      final file = File(imagePath);
      if (!await file.exists()) {
        throw Exception('图片文件不存在: $imagePath');
      }

      // 调试日志：输出文件路径和存在状态
      debugPrint('预测文件路径: $imagePath');
      debugPrint('文件存在状态: ${await file.exists()}');

      final response = await _dio.post(
        '$serverAddress/predict',
        data: FormData.fromMap({
          'username': username,
          'password': password,
          'image': await MultipartFile.fromFile(
            imagePath,
            filename: imagePath.split('/').last,
          ),
        }),
        options: Options(contentType: Headers.multipartFormDataContentType),
      );
      return response.data;
    } catch (e) {
      throw Exception('图片上传失败: $e');
    }
  }
}