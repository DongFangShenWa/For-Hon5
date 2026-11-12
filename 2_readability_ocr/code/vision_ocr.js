ObjC.import("Cocoa");
ObjC.import("Vision");

// 用法: osascript -l JavaScript vision_ocr.js <图片绝对路径>
// 输出: 每行一个识别文本块；失败输出 OCR_FAILED
function run(argv) {
  const path = argv[0];
  const handler = $.VNImageRequestHandler.alloc.initWithURLOptions(
      $.NSURL.fileURLWithPath(path), $.NSDictionary.dictionary);
  const req = $.VNRecognizeTextRequest.alloc.initWithCompletionHandler(() => {});
  req.recognitionLevel = $.VNRequestTextRecognitionLevelAccurate;
  req.usesLanguageCorrection = true;
  const ok = handler.performRequestsError($.NSArray.arrayWithObject(req), null);
  if (!ok) { return "OCR_FAILED"; }
  const obs = req.results;
  const n = typeof obs.count === "function" ? obs.count() : obs.count;
  const lines = [];
  for (let i = 0; i < n; i++) {
    const cand = obs.objectAtIndex(i).topCandidates(1).objectAtIndex(0);
    lines.push(cand.string.js);
  }
  return lines.length ? lines.join("\n") : "OCR_EMPTY";
}
