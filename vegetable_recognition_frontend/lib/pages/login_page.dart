import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../provider/user.dart';
import '../model.dart';
import '../services/user.dart';

class LoginPage extends ConsumerWidget {
  const LoginPage({super.key});


  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final TextEditingController serverAddressController = TextEditingController();
    final TextEditingController usernameController = TextEditingController();
    final TextEditingController passwordController = TextEditingController();

    Future<void> handleAuth(String action) async {
      final prefs = await SharedPreferences.getInstance();
      final userPersistence = ref.read(userPersistenceProvider);

      final serverAddress = serverAddressController.text;
      final user = User(
        username: usernameController.text,
        password: passwordController.text,
      );

      try {
        await prefs.setString('server_address', serverAddress);
        await userPersistence.setUser(user);

        final result = action == 'login'
            ? await UserService().verifyUser(user, serverAddress)
            : await UserService().createUser(user, serverAddress);

        if (result == 'success' && context.mounted) {
          Navigator.pushReplacementNamed(context, '/home');
        }
      } catch (e) {
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('操作失败: $e')),
          );
        }
      }
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('登录'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              decoration: const InputDecoration(
                labelText: '服务器地址',
              ),
              controller: serverAddressController,
            ),
            const SizedBox(height: 16.0),
            TextField(
              decoration: const InputDecoration(
                labelText: '用户名',
              ),
              controller: usernameController,
            ),
            const SizedBox(height: 16.0),
            TextField(
              decoration: const InputDecoration(
                labelText: '密码',
              ),
              obscureText: true,
              controller: passwordController,
            ),
            const SizedBox(height: 32.0),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                ElevatedButton(
                  onPressed: () => handleAuth('login'),
                  child: const Text('登录'),
                ),
                ElevatedButton(
                  onPressed: () => handleAuth('register'),
                  child: const Text('注册'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}