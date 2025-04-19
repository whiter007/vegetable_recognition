import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../widgets/camera.dart';
import '../widgets/response_text.dart';
import '../provider/user.dart';
import '../services/user.dart';
import '../services/predict.dart';

class HomePage extends ConsumerWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final GlobalKey<CameraPreviewWidgetState> cameraKey = GlobalKey();
    return Scaffold(
      appBar: AppBar(
        title: const Text('果蔬识别'),
      ),
      body: Column(
        children: [
          CameraPreviewWidget(
            key: cameraKey,
          ),
          Align(
            alignment: Alignment.centerLeft,
            child: Expanded(
              child: const ResponseTextWidget(),
            ),
          ),
        ],
      ),
      floatingActionButton: Stack(
        children: [
          Positioned(
            bottom: 16.0,
            right: 16.0,
            child: FloatingActionButton(
              onPressed: () async {
                final userPersistence = ref.read(userPersistenceProvider);
                final user = await userPersistence.getUser();
                final serverAddress = await userPersistence.getServerAddress();
                if (user != null && serverAddress != null) {
                  try {
                    await UserService().deleteUser(user, serverAddress);
                    await userPersistence.removeUser();
                    if (context.mounted) {
                      Navigator.pushReplacementNamed(context, '/login');
                    }
                  } catch (e) {
                    if (context.mounted) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('删除失败: $e')),
                      );
                    }
                  }
                }
              },
              backgroundColor: Colors.red,
              mini: true,
              child: const Icon(Icons.close),
            ),
          ),
          Positioned(
            bottom: 16.0,
            left: MediaQuery.of(context).size.width / 2 - 28,
            child: FloatingActionButton(
              onPressed: () async {
                try {
                  final image = await cameraKey.currentState?.takePicture();
                  if (image == null) {
                    throw Exception('未能获取有效图片');
                  }
                  final result = await PredictService().imagePredict(image);
                  ref.read(responseTextProvider.notifier).state = result;
                } catch (e) {
                  if (context.mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('识别失败: $e')),
                    );
                  }
                }
              },
              child: const Icon(Icons.camera),
            ),
          ),
        ],
      ),
    );
  }
}