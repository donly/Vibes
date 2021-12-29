/// Copyright (c) 2021 Razeware LLC
/// 
/// Permission is hereby granted, free of charge, to any person obtaining a copy
/// of this software and associated documentation files (the "Software"), to deal
/// in the Software without restriction, including without limitation the rights
/// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
/// copies of the Software, and to permit persons to whom the Software is
/// furnished to do so, subject to the following conditions:
/// 
/// The above copyright notice and this permission notice shall be included in
/// all copies or substantial portions of the Software.
/// 
/// Notwithstanding the foregoing, you may not use, copy, modify, merge, publish,
/// distribute, sublicense, create a derivative work, and/or sell copies of the
/// Software in any work that is designed, intended, or marketed for pedagogical or
/// instructional purposes related to programming, coding, application development,
/// or information technology.  Permission for such use, copying, modification,
/// merger, publication, distribution, sublicensing, creation of derivative works,
/// or sale is expressly withheld.
/// 
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.

import CoreML

// 映射可更新的模型
struct UpdatableModel {
  private static var updatedDrawingClassifier: UpdatableDrawingClassifier?
  private static let appDirectory = FileManager.default.urls(
    for: .applicationSupportDirectory,
       in: .userDomainMask).first!
  // 指向原始编译模型
  private static let defaultModelURL =
  UpdatableDrawingClassifier.urlOfModelInThisBundle
  // 保存模型的位置
  private static var updatedModelURL =
  appDirectory.appendingPathComponent("personalized.mlmodelc")
  private static var tempUpdatedModelURL =
  appDirectory.appendingPathComponent("personalized_tmp.mlmodelc")
  
  private init() { }
  
  static var imageConstraint: MLImageConstraint {
    guard let model = try? updatedDrawingClassifier ?? UpdatableDrawingClassifier(configuration: MLModelConfiguration()) else {
      fatalError("init UpdatableDrawingClassifier error")
    }
    return model.imageConstraint
  }
}

private extension UpdatableModel {
  // 加载模型
  static func loadModel() {
    let fileManager = FileManager.default
    if !fileManager.fileExists(atPath: updatedModelURL.path) {
      do {
        let updatedModelParentURL =
        updatedModelURL.deletingLastPathComponent()
        try fileManager.createDirectory(
          at: updatedModelParentURL,
          withIntermediateDirectories: true,
          attributes: nil)
        let toTemp = updatedModelParentURL
          .appendingPathComponent(defaultModelURL.lastPathComponent)
        try fileManager.copyItem(
          at: defaultModelURL,
          to: toTemp)
        try fileManager.moveItem(
          at: toTemp,
          to: updatedModelURL)
      } catch {
        print("Error: \(error)")
        return
      }
    }
    guard let model = try? UpdatableDrawingClassifier(
      contentsOf: updatedModelURL) else {
        return
      }
    // 模型加载到内存
    updatedDrawingClassifier = model
  }
  
  static func saveUpdatedModel(_ updateContext: MLUpdateContext) {
    // 首先，从内存中获取更新的模型。这与原始模型不一样。
    let updatedModel = updateContext.model
    let fileManager = FileManager.default
    do {
      // 然后，创建一个中间文件夹来保存更新的模型。
      try fileManager.createDirectory(
          at: tempUpdatedModelURL,
          withIntermediateDirectories: true,
          attributes: nil)
      // 把更新的模型写到一个临时文件夹中
      try updatedModel.write(to: tempUpdatedModelURL)
      // 替换模型文件夹的内容
      // 直接覆盖现有的mlmodelc文件夹会出现错误。
      // 解决方案是保存到一个中间文件夹，然后把内容复制过来。
      _ = try fileManager.replaceItemAt(
        updatedModelURL,
        withItemAt: tempUpdatedModelURL)
      print("Updated model saved to:\n\t\(updatedModelURL)")
    } catch let error {
      print("Could not save updated model to the file system: \(error)")
      return
    }
  }
  
}

extension UpdatableModel {
  static func predictLabelFor(_ value: MLFeatureValue) -> String? {
    loadModel()
    return updatedDrawingClassifier?.predictLabelFor(value)
  }
  
  static func updateWith(
    trainingData: MLBatchProvider,
    completionHandler: @escaping () -> Void
  ) {
    loadModel()
    UpdatableDrawingClassifier.updateModel(
      at: updatedModelURL,
      with: trainingData) { context in
        saveUpdatedModel(context)
        DispatchQueue.main.async { completionHandler() }
    }
  }
}

extension UpdatableDrawingClassifier {
  // 确保图像与模型所期望的一致
  var imageConstraint: MLImageConstraint {
    return model.modelDescription
      .inputDescriptionsByName["drawing"]!
      .imageConstraint!
  }
  
  // 用绘图的CVPixelBuffer调用模型的预测方法。
  // 返回预测的表情符号标签，如果没有匹配的则返回nil
  func predictLabelFor(_ value: MLFeatureValue) -> String? {
    guard
      let pixelBuffer = value.imageBufferValue,
      let prediction = try? prediction(drawing: pixelBuffer).label
    else {
      return nil
    }
    if prediction == "unknown" {
      print("No prediction found")
      return nil
    }
    return prediction
  }
  
  static func updateModel(
    at url: URL,
    with trainingData: MLBatchProvider,
    completionHandler: @escaping (MLUpdateContext) -> Void
  ) {
    do {
      let updateTask = try MLUpdateTask(
        forModelAt: url,
        trainingData: trainingData,
        configuration: nil,
        completionHandler: completionHandler)
      updateTask.resume()
    } catch {
      print("Couldn't create an MLUpdateTask.")
    }
  }
}
