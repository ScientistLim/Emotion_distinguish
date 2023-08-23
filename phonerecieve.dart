import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';

Future<String> analyzeImage(File imageFile) async {
  var uri = Uri.parse("http://YOUR_SERVER_IP:PORT/analyze/");

  var request = http.MultipartRequest('POST', uri)
    ..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

  var response = await request.send();

  if (response.statusCode == 200) {
    var responseBody = await response.stream.bytesToString();
    var jsonData = json.decode(responseBody);
    return jsonData['emotion'];
  } else {
    throw Exception('Failed to analyze image');
  }
}
