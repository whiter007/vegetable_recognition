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
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    if (!mounted) return;

    try {
      final cameras = await availableCameras();
      final firstCamera = cameras.first;

      _controller = CameraController(
        firstCamera,
        ResolutionPreset.max,
      );

      _initializeControllerFuture = _controller.initialize().then((_) async {
        if (!mounted) {
          await _controller.dispose();
          return;
        }
        await _controller.setFlashMode(FlashMode.off);
        await _controller.lockCaptureOrientation(DeviceOrientation.portraitUp);
        await _controller.setFocusMode(FocusMode.auto);
        if (mounted) setState(() {});
      });

    } catch (e) {
      debugPrint('相机初始化失败: $e');
      if (_controller.value.isInitialized) {
        await _controller.dispose();
      }
      _initializeControllerFuture = Future.value();
      if (mounted) {
        setState(() => _initializeControllerFuture = Future.error(e));
      }
    }
  }

  Future<String> takePicture() async {
    try {
      await _initializeControllerFuture;
      final image = await _controller.takePicture();
      return image.path;
    } catch (e) {
      debugPrint('拍照失败: $e');
      rethrow;
    }
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
      future: _initializeControllerFuture,
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return Center(child: Text('相机初始化失败'));
        }
        if (snapshot.connectionState == ConnectionState.done) {
          return _controller.value.isInitialized
              ? LayoutBuilder(
                  builder: (context, constraints) {
                    final aspectRatio = _controller.value.aspectRatio > 1
                        ? 1 / _controller.value.aspectRatio
                        : _controller.value.aspectRatio;

                    return Expanded(
                      child: AspectRatio(
                        aspectRatio: aspectRatio,
                        child: CameraPreview(_controller),
                      ),
                    );
                  },
                )
              : const Center(child: Text('相机不可用'));
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