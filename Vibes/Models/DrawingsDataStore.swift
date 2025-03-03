/// Copyright (c) 2020 Razeware LLC
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
/// This project and source code may use libraries or frameworks that are
/// released under various Open-Source licenses. Use of those libraries and
/// frameworks are governed by their own individual licenses.
///
/// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
/// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
/// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
/// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
/// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
/// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
/// THE SOFTWARE.

import CoreML

class DrawingDataStore: NSObject {
  // MARK: - Properties
  private var drawings: [Drawing?]
  
  let emoji: String
  
  var numberOfDrawings: Int {
    return drawings.filter({ $0 != nil }).count
  }
  
  init(for emoji: String, capacity: Int) {
    self.emoji = emoji
    self.drawings = Array(repeating: nil, count: capacity)
  }
}

// MARK: - Helper methods
extension DrawingDataStore {
  func addDrawing(_ drawing: Drawing, at index: Int) {
    drawings[index] = drawing
  }
  
  func prepareTrainingData() throws -> MLBatchProvider {
    // 初始化一个空的 MLFeatureProvider 数组
    var featureProviders: [MLFeatureProvider] = []
    // 定义模型训练输入的名称
    let inputName = "drawing"
    let outputName = "label"
    // 循环浏览数据存储中的图形数据
    for drawing in drawings {
      if let drawing = drawing {
        // 将绘图训练输入包在一个特征值中
        let inputValue = drawing.featureValue
        // 将emoji训练输入包在一个特征值中
        let outputValue = MLFeatureValue(string: emoji)
        // 为训练输入创建一个MLFeatureValue集合。这是一个训练输入名称和特征值的字典。
        let dataPointFeatures: [String: MLFeatureValue] =
          [inputName: inputValue,
          outputName: outputValue]
        // 为 MLFeatureValue 集合创建一个 MLFeatureProvider，并将其追加到数组中
        if let provider =
          try? MLDictionaryFeatureProvider(
            dictionary: dataPointFeatures) {
          featureProviders.append(provider)
        }
      }
    }
    // 最后，从MLFeatureProvider数组中创建一个批处理对象（MLArrayBatchProvider）
    return MLArrayBatchProvider(array: featureProviders)
  }
}
