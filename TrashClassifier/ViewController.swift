//
//  ViewController.swift
//  TrashClassifier
//
//  Created by Anton Poluianov on 18/05/2019.
//  Copyright © 2019 Anton Poluianov. All rights reserved.
//

import UIKit
import AVKit
import Vision

enum Trash: String {
	case cardboard = "cardboard"
	case glass = "glass"
	case metal = "metal"
	case paper = "paper"
	case plastic = "plastic"
	case trash = "trash"
}

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
	
	@IBOutlet weak var predictionLabel: UILabel!
    @IBOutlet weak var segmentedControl: UISegmentedControl!
    var model: MLModel = TrashClassifier_820_nocups_1().model

	override func viewDidLoad() {
		super.viewDidLoad()
		setupCamera()
	}
	
	func setupCamera() {
		//Start capture session
		let captureSession = AVCaptureSession()
		captureSession.sessionPreset = .photo
		captureSession.startRunning()
		
		// Add input for capture
		guard let captureDevice = AVCaptureDevice.default(for: .video) else { return }
		guard let captureInput = try? AVCaptureDeviceInput(device: captureDevice) else { return }
		captureSession.addInput(captureInput)
		
		// Add preview layer
		let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
		view.layer.addSublayer(previewLayer)
		previewLayer.frame = view.frame
		
		// Add output for capture
		let dataOutput = AVCaptureVideoDataOutput()
		dataOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
		captureSession.addOutput(dataOutput)
	}
	
	// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
	
	func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
		
		// Initialise CVPixelBuffer from sample buffer
		guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
		
		//Initialise Core ML model
		guard let trashModel = try? VNCoreMLModel(for:self.model) else { return }
		
		// Create a Core ML Vision request
		let request =  VNCoreMLRequest(model: trashModel) { (finishedRequest, err) in
			
			// Dealing with the result of the Core ML Vision request
			guard let results = finishedRequest.results as? [VNClassificationObservation] else { return }
			
			guard let firstResult = results.first else { return }
			var predictionString = ""
			DispatchQueue.main.async {
				switch firstResult.identifier {
				case Trash.cardboard.rawValue:
					predictionString = "cardboard"
				case Trash.glass.rawValue:
					predictionString = "glass"
				case Trash.metal.rawValue:
					predictionString = "metal"
				case Trash.paper.rawValue:
					predictionString = "paper"
				case Trash.plastic.rawValue:
					predictionString = "plastic"
				case Trash.trash.rawValue:
					predictionString = "trash"
				default:
					break
				}
				self.predictionLabel.text = predictionString + "(\(firstResult.confidence))"
			}
		}
		
		// Perform the above request using Vision Image Request Handler
		try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([request])
	}
    
    @IBAction func valuChanged(_ sender: Any) {
        let controlValue = self.segmentedControl.selectedSegmentIndex
        switch controlValue {
        case 0:
            self.model = TrashClassifier_820_nocups_1().model
        case 1:
            self.model = TrashClassifier79accNotAugmented().model
        case 2:
            self.model = TrashClassifierWithLabels88acc().model
        case 3:
            self.model = TrashClassifierWithLabels().model
        default:
            self.model = TrashClassifier_820_nocups_1().model
        }
    }
}

