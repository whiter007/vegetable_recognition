import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'pages/home_page.dart';
import 'pages/login_page.dart';
import 'provider/user.dart';
import 'model.dart';

void main() {
  runApp(
    ProviderScope(
      child: MyApp(),
    ),
  ); // 使用riverpod的ProviderScope提供全局状态管理
}

class MyApp extends ConsumerWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userPersistence = ref.read(userPersistenceProvider);

    return FutureBuilder<User?>(
      future: userPersistence.getUser(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return MaterialApp(
            home: Scaffold(
              body: Center(
                child: CircularProgressIndicator(),
              ),
            ),
          );
        }

        final user = snapshot.data;
        return MaterialApp(
          title: 'Vegetable Recognition',
          theme: ThemeData(
            primarySwatch: Colors.green,
          ),
          home: user == null ? const LoginPage() : const HomePage(),
          routes: {
            '/login': (context) => const LoginPage(),
            '/home': (context) => const HomePage(),
          },
        );
      },
    );
  }
}