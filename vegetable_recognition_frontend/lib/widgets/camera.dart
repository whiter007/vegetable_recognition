import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';

class CameraPreviewWidget extends StatefulWidget {
  final VoidCallback? onCapturePressed;

  const CameraPreviewWidget({super.key, this.onCapturePressed});

  @override
  State<CameraPreviewWidget> createState() => CameraPreviewWidgetState();
}

class CameraPreviewWidgetState extends State<CameraPreviewWidget> {
  CameraController? _controller;
  Future<void>? _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initializeCamera() async {
    // 关键点1：先释放旧控制器
    if (_controller != null) {
      await _controller!.dispose();
      _controller = null;
    }

    if (!mounted) return;

    try {
      final cameras = await availableCameras();
      final firstCamera = cameras.first; // 移除空校验，保留原有逻辑

      final newController = CameraController(
        firstCamera,
        ResolutionPreset.max,
      );

      // 关键点2：异步操作前检查mounted
      if (!mounted) {
        await newController.dispose();
        return;
      }

      _controller = newController;
      _initializeControllerFuture = _controller!.initialize().then((_) async {
        if (!mounted || _controller == null) return;

        await _controller!.setFlashMode(FlashMode.off);
        await _controller!.lockCaptureOrientation(DeviceOrientation.portraitUp);
        await _controller!.setFocusMode(FocusMode.auto);

        if (mounted) setState(() {});
      });

    } catch (e) {
      debugPrint('初始化失败: $e');
      await _controller?.dispose();
      _controller = null;
      _initializeControllerFuture = Future.error(e);
      if (mounted) setState(() {});
    }
  }

  Future<String> takePicture() async {
    try {
      // 关键点1：等待初始化完成
      await _initializeControllerFuture;

      // 关键点2：双重空安全校验
      if (_controller == null || !_controller!.value.isInitialized) {
        throw StateError('相机未就绪');
      }

      // 关键点3：捕获拍照过程中的异常
      final image = await _controller!.takePicture();
      return image.path;
    } on CameraException catch (e) {
      debugPrint('拍照失败: ${e.description}');
      throw Exception('拍照失败: ${e.code}');
    } catch (e) {
      debugPrint('未知错误: $e');
      throw Exception('拍照系统错误');
    }
  }
  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
      future: _initializeControllerFuture,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return const Center(child: Text('摄像头不可用')); // 统一错误提示
        }
      if (snapshot.connectionState == ConnectionState.done) {
        final controller = _controller;
        if (controller != null && controller.value.isInitialized) {
          return LayoutBuilder(
            builder: (context, constraints) {
              // 恢复原始比例计算逻辑
              final rawAspectRatio = controller.value.aspectRatio;
              final aspectRatio = rawAspectRatio > 1
                  ? 1 / rawAspectRatio
                  : rawAspectRatio;

              return Expanded(
                child: AspectRatio(
                  aspectRatio: aspectRatio,
                  child: CameraPreview(controller),
                ),
              );
            },
          );
        }
        return const Center(child: Text('控制器未初始化'));
      }
        return const Center(child: CircularProgressIndicator());
      },
    );
  }
}

class SquareCenterClipper extends CustomClipper<Rect> {
  @override
  Rect getClip(Size size) {
    final side = size.width < size.height ? size.width : size.height;
    return Rect.fromCenter(
      center: Offset(size.width/2, size.height/2),
      width: side,
      height: side,
    );
  }

  @override
  bool shouldReclip(covariant CustomClipper<Rect> oldClipper) => false;
}