var express = require('express');
var router = express.Router();
const fs = require('fs')

model = JSON.parse(fs.readFileSync('./public/model/model_meta.json', 'utf-8'))

/* GET home page. */
router.get('/', function(req, res, next) {
  let listPoses = model.outputs[0].uniqueValues
  res.render('index', {listPoses: listPoses});
});

module.exports = router;