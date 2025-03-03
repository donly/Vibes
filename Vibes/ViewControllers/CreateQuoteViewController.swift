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

import UIKit
import Vision

class CreateQuoteViewController: UIViewController {
  // MARK: - Properties
  @IBOutlet weak var imageView: UIImageView!
  @IBOutlet weak var quoteTextView: UITextView!
  @IBOutlet weak var addStickerButton: UIBarButtonItem!
  @IBOutlet weak var stickerView: UIView!
	@IBOutlet weak var starterLabel: UILabel!
	
  var drawingView: DrawingView!    // 用户绘制快捷方式的画布视图
  
  private lazy var quoteList: [Quote] = {
    guard let path = Bundle.main.path(forResource: "Quotes", ofType: "plist")
      else {
        print("Failed to read Quotes.plist")
        return []
    }
    let fileUrl = URL.init(fileURLWithPath: path)
    guard let quotesArray = NSArray(contentsOf: fileUrl) as? [Dictionary<String, Any>]
      else { return [] }
    let quotes: [Quote] = quotesArray.compactMap { (quote) in
      guard
        let text = quote[Quote.Key.text] as? String,
        let author = quote[Quote.Key.author] as? String,
        let keywords = quote[Quote.Key.keywords] as? [String]
        else { return nil }
      
      return Quote(
        text: text,
        author: author,
        keywords: keywords)
    }
    return quotes
  }()
  
  private lazy var stickerFrame: CGRect = {
    let stickerHeightWidth = 50.0
    let stickerOffsetX =
      Double(stickerView.bounds.midX) - (stickerHeightWidth / 2.0)
    let stickerRect = CGRect(
      x: stickerOffsetX,
      y: 80.0, width:
      stickerHeightWidth,
      height: stickerHeightWidth)
    return stickerRect
  }()
  
  private lazy var classificationRequest: VNCoreMLRequest = {
    do {
      let model = try VNCoreMLModel(for: SqueezeNet(configuration: MLModelConfiguration()).model)
      let request = VNCoreMLRequest(model: model) { [weak self] request, error in
        guard let self = self else { return }
        // 当预测结束时，调用方法更新箴言
        self.processClassifications(for: request, error: error)
      }
      request.imageCropAndScaleOption = .centerCrop
      return request
    } catch {
      fatalError("Failed to load Vision ML model: \(error)")
    }
  }()
  
  // MARK: - Lifecycle
  override func viewDidLoad() {
    super.viewDidLoad()
    quoteTextView.isHidden = true
    addStickerButton.isEnabled = false
    
    // 添加绘图视图，初始隐藏
    addCanvasForDrawing()
    drawingView.isHidden = true
  }
  
  // MARK: - Actions
  @IBAction func selectPhotoPressed(_ sender: Any) {
    let picker = UIImagePickerController()
    picker.delegate = self
    picker.sourceType = .photoLibrary
    picker.modalPresentationStyle = .overFullScreen
    present(picker, animated: true)
  }
  
  @IBAction func cancelPressed(_ sender: Any) {
    dismiss(animated: true)
  }
  
  @IBAction func addStickerDoneUnwind(_ unwindSegue: UIStoryboardSegue) {
    guard
      let sourceViewController = unwindSegue.source as? AddStickerViewController,
      let selectedEmoji = sourceViewController.selectedEmoji
      else {
        return
    }
    addStickerToCanvas(selectedEmoji, at: stickerFrame)
  }
}

// MARK: - Private methods
private extension CreateQuoteViewController {
  func addStickerToCanvas(_ sticker: String, at rect: CGRect) {
    let stickerLabel = UILabel(frame: rect)
    stickerLabel.text = sticker
    stickerLabel.font = .systemFont(ofSize: 100)
    stickerLabel.numberOfLines = 1
    stickerLabel.baselineAdjustment = .alignCenters
    stickerLabel.textAlignment = .center
    stickerLabel.adjustsFontSizeToFitWidth = true
    
    // Add gesture recognizer
//    stickerLabel.isUserInteractionEnabled = true
//    let panGestureRecognizer = UIPanGestureRecognizer(
//      target: self,
//      action: #selector(handlePanGesture(_:)))
//    stickerLabel.addGestureRecognizer(panGestureRecognizer)
    
    // Add sticker to the canvas
    stickerView.addSubview(stickerLabel)
  }
  
  func clearStickersFromCanvas() {
    for view in stickerView.subviews {
      view.removeFromSuperview()
    }
  }
  
  func addCanvasForDrawing() {
    // 创建绘图视图示例
    drawingView = DrawingView(frame: stickerView.bounds)
    drawingView.delegate = self

    // 添加到主视图
    view.addSubview(drawingView)
    // 添加约束防止和贴纸视图重叠
    drawingView.translatesAutoresizingMaskIntoConstraints = false
    NSLayoutConstraint.activate([
      drawingView.topAnchor.constraint(equalTo: stickerView.topAnchor),
      drawingView.leftAnchor.constraint(equalTo: stickerView.leftAnchor),
      drawingView.rightAnchor.constraint(equalTo: stickerView.rightAnchor),
      drawingView.bottomAnchor.constraint(equalTo: stickerView.bottomAnchor)
    ])
  }
  
  func getQuote(for keywords: [String]? = nil) -> Quote? {
    if let keywords = keywords {
      for keyword in keywords {
        for quote in quoteList {
          if quote.keywords.contains(keyword) {
            return quote
          }
        }
      }
    }
    return selectRandomQuote()
  }
  
  func selectRandomQuote() -> Quote? {
    if let quote = quoteList.randomElement() {
      return quote
    }
    return nil
  }
  
  func classifyImage(_ image: UIImage) {
    guard let orientation = CGImagePropertyOrientation(rawValue: UInt32(image.imageOrientation.rawValue)) else { return }
    guard let ciImage = CIImage(image: image) else {
        fatalError("Unable to create \(CIImage.self) from \(image).")
      }
    
    // 在一个后台队列中启动一个异步分类请求，当句柄在外部被创建并且安排上时，这个Vision请求将被执行
    DispatchQueue.global(qos: .userInitiated).async {
      let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
      do {
        try handler.perform([self.classificationRequest])
      } catch {
        print("Failed to perform classification.\n\(error.localizedDescription)")
      }
    }
  }
  
  func processClassifications(for request: VNRequest, error: Error?) {
    DispatchQueue.main.async {
      // 处理来自图像分类请求的结果
      if let classifications =
          request.results as? [VNClassificationObservation] {
        // 提取前两个预测结果
        let topClassifications = classifications.prefix(2).map {
          (confidence: $0.confidence, identifier: $0.identifier)
        }
        print("Top classifications: \(topClassifications)")
        let topIdentifiers =
        topClassifications.map {$0.identifier.lowercased() }
        // 将预测结果传入getQuote(for:)获得一个相关的箴言
        if let quote = self.getQuote(for: topIdentifiers) {
          self.quoteTextView.text = quote.text
        }
      }
    }
  }
}

// MARK: - UIImagePickerControllerDelegate
extension CreateQuoteViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
  func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
    picker.dismiss(animated: true)
    
    let image = info[UIImagePickerController.InfoKey.originalImage] as! UIImage
    imageView.image = image
    quoteTextView.isHidden = false
    addStickerButton.isEnabled = true
    // 清除画布，隐藏绘图视图以便表情符号的贴纸可以正确展示
//    drawingView.clearCanvas()
    drawingView.isHidden = false
		starterLabel.isHidden = true
    clearStickersFromCanvas()
    
//    if let quote = getQuote() {
//      quoteTextView.text = quote.text
//    }
    
    classifyImage(image)
  }
}

// MARK - UIGestureRecognizerDelegate
extension CreateQuoteViewController: UIGestureRecognizerDelegate {
  @objc func handlePanGesture(_ recognizer: UIPanGestureRecognizer) {
    let translation = recognizer.translation(in: stickerView)
    if let view = recognizer.view {
      view.center = CGPoint(
        x:view.center.x + translation.x,
        y:view.center.y + translation.y)
    }
    recognizer.setTranslation(CGPoint.zero, in: stickerView)
    
    if recognizer.state == UIGestureRecognizer.State.ended {
        let velocity = recognizer.velocity(in: stickerView)
        let magnitude =
          sqrt((velocity.x * velocity.x) + (velocity.y * velocity.y))
        let slideMultiplier = magnitude / 200
          
        let slideFactor = 0.1 * slideMultiplier
        var finalPoint = CGPoint(
          x:recognizer.view!.center.x + (velocity.x * slideFactor),
          y:recognizer.view!.center.y + (velocity.y * slideFactor))
        finalPoint.x =
          min(max(finalPoint.x, 0), stickerView.bounds.size.width)
        finalPoint.y =
          min(max(finalPoint.y, 0), stickerView.bounds.size.height)
          
        UIView.animate(
          withDuration: Double(slideFactor * 2),
          delay: 0,
          options: UIView.AnimationOptions.curveEaseOut,
          animations: {recognizer.view!.center = finalPoint },
          completion: nil)
    }
  }
}

extension CreateQuoteViewController: DrawingViewDelegate {
  func drawingDidChange(_ drawingView: DrawingView) {
    // 绘图的边界，防止越界
    let drawingRect = drawingView.boundingSquare()
    // 绘图实例
    let drawing = Drawing(
      drawing: drawingView.canvasView.drawing,
      rect: drawingRect)
    // 为绘图预测输入创建特征值
    let imageFeatureValue = drawing.featureValue
    // 进行预测，以获得与该绘制图形相对应的表情符号
    let drawingLabel =
      UpdatableModel.predictLabelFor(imageFeatureValue)
    // 更新主队列中的视图，清除画布并将预测的表情符号添加到主视图中
    DispatchQueue.main.async {
      drawingView.clearCanvas()
      guard let emoji = drawingLabel else {
        return
      }
      self.addStickerToCanvas(emoji, at: drawingRect)
    }
  }
}
