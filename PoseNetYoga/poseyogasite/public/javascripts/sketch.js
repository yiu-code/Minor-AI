let video;
let poseNet;
let pose;
let skeleton;

let brain;
let poseLabel = "";

let targetLabel;
let currentButton = null;

let startLogging = false;

function setup() {
  canvas = createCanvas(640, 480);
  canvas.parent('videoContainer');
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video);
  poseNet.on('pose', gotPoses);

  let options = {
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  // LOAD TRAINING DATA
  // brain.loadData('../javascripts/PoseNetV2.json', dataReady);

  
   //LOAD PRETRAINED MODEL
   //Uncomment to train your own model!
  const modelInfo = {
	model: 'model/model.json',
	metadata: 'model/model_meta.json',
	weights: 'model/model.weights.bin',
   };
  brain.load(modelInfo, classifyPose());

  createButtons();
}

function dataReady() {
  brain.normalizeData();
  brain.train({
    epochs: 100,
    batchsize: 24
  }, classifyPose);
}

function keyPressed() {
  if (key == 't') {
    setTimeout(function () {
      startLogging = true;
      setTimeout(function () {
        startLogging = false;
      }, 3000)
    }, 5000)
  }
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(_, results) {
  if (startLogging) {
    console.log(results);
  }
 
  if (results[0].label == currentButton && results[0].confidence > 0.70){
	poseLabel = results[0].label;
  }
  else {
		poseLabel = currentButton ? 'No ' + currentButton : 'No Pose Selected'
  }

  classifyPose();
}

function createButtons() {
  $.getJSON('model/model_meta.json', function (data) {
    listPoses = data.outputs[0].uniqueValues;
    for (i = 0; i < listPoses.length; i++){
        createNewButton(listPoses[i])
    };

  }); // Calling upon all present poses in the JSON file

}

function createNewButton(elemId) {
  button = document.getElementById(elemId);
  button.addEventListener('click', function(e){
    console.log(elemId+ ", activate!");
    currentButton = elemId;
  });

}

function finished() {
  console.log('model trained');
  brain.save(); //UNCOMMENT WHEN YOU WANT TO SAVE THE MODEL
  classifyPose();
}

function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
}

function draw() {
  push();
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(0);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0);
      stroke(255);
      ellipse(x, y, 16, 16);
    }
  }
  pop();

  fill(255, 0, 255);
  noStroke();
  textSize(50);
  textAlign(CENTER, CENTER);
  text(poseLabel, width / 2, height / 2);
}