import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

final responseTextProvider = StateProvider<String>((ref) => '服务器输出文本');

class ResponseTextWidget extends ConsumerWidget {
  const ResponseTextWidget({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final responseText = ref.watch(responseTextProvider);
    return Expanded(
      child: Text(responseText),
    );
  }
}