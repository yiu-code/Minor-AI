var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  let listPoses = callJson();
  res.render('index', {listPoses: listPoses});
});

module.exports = router;
