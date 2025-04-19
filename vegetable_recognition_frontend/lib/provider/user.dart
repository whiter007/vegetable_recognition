import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../model.dart';

class UserPersistence {
  Future<void> setUser(User user) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('username', user.username);
    await prefs.setString('password', user.password);
  }

  Future<User?> getUser() async {
    final prefs = await SharedPreferences.getInstance();
    final username = prefs.getString('username');
    final password = prefs.getString('password');
    if (username == null || password == null) {
      return null;
    }
    return User(username: username, password: password);
  }

  Future<void> removeUser() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('username');
    await prefs.remove('password');
    await prefs.remove('server_address');
  }

  Future<String?> getServerAddress() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString('server_address');
  }
}

final userPersistenceProvider = Provider<UserPersistence>((ref) {
  return UserPersistence();
});

