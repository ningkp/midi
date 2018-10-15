// var leftTopBox = new Vue({
//     el: '#leftTopBox',
//     data: {
//         login: false,
//         userdata: Object,
//         name: null,
//         passerror: false,
//     }
// });

var login_right_top = new Vue({
	el: '#right-top',
	data: {
		login: false,
	}
});

var noLogin = new Vue({
	el: '#unlogin',
	data:{
		login: true,
	}
});

 //Activity_video
 var nuaa_Activity = new Vue({
    el: "#nuaa_Activity",
    data:{
        bigOne: Object(),
        big_Small: Array(),
        smallOne: Array(),
         totalVideo : Array(), 
    }
 });

 //Nature_video
 var nuaa_Nature = new Vue({
    el: "#nuaa_Nature",
    data:{
        bigOne: Array(),
        big_Small: Array(),
        smallOne: Array(),
         totalVideo : Array(), 
    }
 });

 //Street_video
 var nuaa_Street = new Vue({
    el: "#nuaa_Street",
    data:{
        bigOne: Object(),
        big_Small: Array(),
        smallOne: Array(),
         totalVideo : Array(), 
    }
 });

 //Rocket_video
var nuaa_Rocket = new Vue({
    el: "#nuaa_Rocket",
    data:{
        bigOne: Object(),
        big_Small: Array(),
        smallOne: Array(),
         totalVideo : Array(), 
    }
 });

 //Rocket_video
var nuaa_About = new Vue({
    el: "#nuaa_About",
    data:{
        bigOne: Object(),
        big_Small: Array(),
        smallOne: Array(),
         totalVideo : Array(), 
    }
 });

function getVideoByClassifyToIndex (classify){
    var classifyData = {"classify":classify};
    $.ajax({                                    //获取视频数据
        url: "?m=video&c=index&a=AJgetVideoByClassify",
        type: "POST",
        data: classifyData,
        error: function(request) {
            // console.log(request);
            console.log('error');
        },
        success: function(data) {
            //console.log(data); 
            data = JSON.parse(data);
            //console.log(data); 
            displayIndex(data);
        }
    });
}

function displayIndex(data){
    var activityNum = 0;
    var activityVideo = Array();
    var natureNum = 0;
    var natureVideo = Array();
    var streetNum = 0;
    var streetVideo = Array();
    var rocketNum = 0;
    var rocketVideo = Array();
    if(data[0].classify==0){        //Activity_video
        for(var i in data){
            activityVideo[activityNum] = data[i];
            activityVideo[activityNum].picSrc = "./Uploads/photo/video/" + data[i].vid + ".jpg";
            activityVideo[activityNum].videoSrc = "?m=video&c=Index&a=play&vid="+ data[i].classify + "_" + data[i].vid;
            if(activityNum<9){
                activityVideo[activityNum].isShow = true;
            }
            else{
                activityVideo[activityNum].isShow = false;
            }
            activityNum++;
        }
    }
    else if(data[0].classify==1){        //Activity_video
        for(var i in data){
            natureVideo[natureNum] = data[i];
            natureVideo[natureNum].picSrc = "./Uploads/photo/video/" + data[i].vid + ".jpg";
            natureVideo[natureNum].videoSrc = "?m=video&c=Index&a=play&vid="+ data[i].classify + "_" + data[i].vid;
            if(natureNum<8){
                natureVideo[natureNum].isShow = true;
            }
            else{
                natureVideo[natureNum].isShow = false;
            }
            natureNum++;
        }
    }
    else if(data[0].classify==2){
        for(var i in data){
            streetVideo[streetNum] = data[i];
            streetVideo[streetNum].picSrc = "./Uploads/photo/video/" + data[i].vid + ".jpg";
            streetVideo[streetNum].videoSrc = "?m=video&c=Index&a=play&vid="+ data[i].classify + "_" + data[i].vid;
            if(streetNum<8){
                streetVideo[streetNum].isShow = true;
            }
            else{
                streetVideo[streetNum].isShow = false;
            }
            streetNum++;       
        }
    }
    else if(data[0].classify==3){
        for(var i in data){
            rocketVideo[rocketNum] = data[i];
            rocketVideo[rocketNum].picSrc = "./Uploads/photo/video/" + data[i].vid + ".jpg";
            rocketVideo[rocketNum].videoSrc = "?m=video&c=Index&a=play&vid="+ data[i].classify + "_" + data[i].vid;
            if(rocketNum<8){
                rocketVideo[rocketNum].isShow = true;
            }
            else{
                rocketVideo[rocketNum].isShow = false;
            }
            rocketNum++;    
        }   
    }
    if(data[0].classify==0){
        //nuaa_Activity.totalVideo = activityVideo;
        var j = 0;
        var k = 0;
        var big_Small = Array();
        var smallOne = Array();
        for(i=1;i<3;i++,j++){
            big_Small[j] = activityVideo[i];
        }
        for(i=3;i<9;i++,k++){
            smallOne[k] = activityVideo[i];
        }
        nuaa_Activity.bigOne = activityVideo[0];
        nuaa_Activity.big_Small = big_Small;
        nuaa_Activity.smallOne = smallOne;
        //console.log(nuaa_Activity);
    }
    else if(data[0].classify==1){
        //nuaa_Nature.totalVideo = natureVideo;
        var j = 0;
        var k = 0;
        var big_Small = Array();
        var smallOne = Array();
        for(i=1;i<3;i++,j++){
            big_Small[j] = natureVideo[i];
        }
        for(i=3;i<9;i++,k++){
            smallOne[k] = natureVideo[i];
        }
        nuaa_Nature.bigOne = natureVideo[0];
        nuaa_Nature.big_Small = big_Small;
        nuaa_Nature.smallOne = smallOne;
        //console.log(nuaa_Nature);
    }
    else if(data[0].classify==2){
        //nuaa_Street.totalVideo = streetVideo;
        var j = 0;
        var k = 0;
        var big_Small = Array();
        var smallOne = Array();
        for(i=1;i<3;i++,j++){
            big_Small[j] = streetVideo[i];
        }
        for(i=3;i<9;i++,k++){
            smallOne[k] = streetVideo[i];
        }
        nuaa_Street.bigOne = streetVideo[0];
        nuaa_Street.big_Small = big_Small;
        nuaa_Street.smallOne = smallOne;
        //console.log(nuaa_Street);
    }
    else if(data[0].classify==3){
        //nuaa_Rocket.totalVideo = rocketVideo;
        var j = 0;
        var k = 0;
        var big_Small = Array();
        var smallOne = Array();
        for(i=1;i<3;i++,j++){
            big_Small[j] = rocketVideo[i];
        }
        for(i=3;i<9;i++,k++){
            smallOne[k] = rocketVideo[i];
        }
        nuaa_Rocket.bigOne = rocketVideo[0];
        nuaa_Rocket.big_Small = big_Small;
        nuaa_Rocket.smallOne = smallOne;
        //console.log(nuaa_Rocket);
    }
}

 $(document).ready(function(){
    $.ajax({
        type: "POST",
        url: '?m=user&c=index&a=AJcheckLogin',
        error: function(request) {
            // console.log(request);
            console.log('error');
        },
        success: function(data) {
        	data = JSON.parse(data);
            console.log(data); 
            console.log(data.login);
            if(data.login=='1'){
            	noLogin.login = false;
                login_right_top.login = true;
            }    
        }
    });
    getVideoByClassifyToIndex("0");
    getVideoByClassifyToIndex("1");
    getVideoByClassifyToIndex("2");
    getVideoByClassifyToIndex("3"); 
});




