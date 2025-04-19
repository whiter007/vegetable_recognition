import 'package:dio/dio.dart';
import '../model.dart';

class UserService {
  final Dio _dio = Dio();

  Future<String> createUser(User user, String serverAddress) async {
    try {
      final response = await _dio.post(
        '$serverAddress/user/create',
        data: FormData.fromMap({
          'username': user.username,
          'password': user.password,
        }),
        options: Options(contentType: Headers.formUrlEncodedContentType),
      );
      return response.data;
    } catch (e) {
      throw Exception('Failed to create user: $e');
    }
  }

  Future<String> verifyUser(User user, String serverAddress) async {
    try {
      final response = await _dio.post(
        '$serverAddress/user/verify',
        data: FormData.fromMap({
          'username': user.username,
          'password': user.password,
        }),
        options: Options(contentType: Headers.formUrlEncodedContentType),
      );
      return response.data;
    } catch (e) {
      throw Exception('Failed to verify user: $e');
    }
  }

  Future<String> deleteUser(User user, String serverAddress) async {
    try {
      final response = await _dio.post(
        '$serverAddress/user/delete',
        data: FormData.fromMap({
          'username': user.username,
          'password': user.password,
        }),
        options: Options(contentType: Headers.formUrlEncodedContentType),
      );
      return response.data;
    } catch (e) {
      throw Exception('Failed to delete user: $e');
    }
  }
}