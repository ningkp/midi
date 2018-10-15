 //获取地址栏参数
function getQueryString(name) {  
    var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)", "i");  
    var r = window.location.search.substr(1).match(reg);  
    if (r != null) return unescape(r[2]);  
    return null;  
} 

var login_right_top = new Vue({
	el: '#right-top',
	data: {
		login: false,
	}
});

 //学院title
 var collegeTitle = new Vue({
 	el: '#collegeTitle',
 	data: {
 		name: null,
 	}
 });

//New_video
 var New_video = new Vue({
 	el: "#New_video",
 	data:{
         totalVideo : Array(), 
    }
 });


 //Activity_video
 var Activity_video = new Vue({
 	el: "#Activity_video",
 	data:{
         totalVideo : Array(), 
    }
 });

 //Nature_video
 var Nature_video = new Vue({
 	el: "#Nature_video",
 	data:{
         totalVideo : Array(), 
    }
 });

 //Street_video
 var Street_video = new Vue({
 	el: "#Street_video",
 	data:{
         totalVideo : Array(), 
    }
 });

 //Rocket_video
var Rocket_video = new Vue({
	el: "#Rocket_video",
	data:{
	     totalVideo : Array(), 
	}
 });


var flag = true;			//判断more是展开还是折叠

$(document).ready(function() {
	getVideo();

	isLogin();
    
    //more按钮的事件
    $('.moreVideo').click(showAllVideo);

    //下一页按钮事件
    $('.arrow .nextPage').click(showNextPageVideo);

    //上一页按钮事件
    $('.arrow .lastPage').click(showLastPageVideo); 
});

function showAllVideo(){			//展开所有视频或折叠
//	console.log($(this).attr('class'));
	var classify = $(this).attr('class').split(" ");
	classify = classify[1] + "_video";					//得到要操作的类
	switch(classify){
		case "Activity_video":{
			classify = Activity_video;
			break;
		}
		case "Nature_video":{
			classify = Nature_video;
			break;
		}
		case "Street_video":{
			classify = Street_video;
			break;
		}
		case "Rocket_video":{
			classify = Rocket_video;
			break;
		}
		case "New_video":{						//该功能暂时未上线
			classify = New_video;
			break;
		}
	}
//	console.log(classify);
	if(flag)					//展开
	{
		for(var i in classify.totalVideo){
			classify.totalVideo[i].isShow = true;
		}
		flag = false;
	}	
	else{
		for(var i in classify.totalVideo){
			if(i<8){
				classify.totalVideo[i].isShow = true;
			}
			else{
				classify.totalVideo[i].isShow = false;
			}
		}
		flag = true;
	}
}

function showNextPageVideo(){			//显示下一页视频
	var classify = $(this).parent().attr('class').split(" ");
	classify = classify[1] + "_video";					//得到要操作的类
	switch(classify){
		case "Activity_video":{
			classify = Activity_video;
			break;
		}
		case "Nature_video":{
			classify = Nature_video;
			break;
		}
		case "Street_video":{
			classify = Street_video;
			break;
		}
		case "Rocket_video":{
			classify = Rocket_video;
			break;
		}
		case "New_video":{						//该功能暂时未上线
			classify = New_video;
			break;
		}
	}
	console.log(classify);
	var isNeed = false;			//是否需要翻页
	var num = 0;
	for(var i in classify.totalVideo){
		if(classify.totalVideo[i].isShow==true&&isNeed==false){		//找到第一个isShow为true且判断是否需要翻页
			if(classify.totalVideo[parseInt(i)+8].isShow==false)			
			{
				isNeed = true;
			}
		}
		if(isNeed==true&&num<=8)
		{
			classify.totalVideo[i].isShow = false;
			num++;
		}
		if(num>8&&num<=16){
			classify.totalVideo[i].isShow = true;
			num++;
		}
		if(num>16)
		{
			break;
		}
	}
}
function showLastPageVideo() {					//显示上一页视频
	var classify = $(this).parent().attr('class').split(" ");
	classify = classify[1] + "_video";					//得到要操作的类
	switch(classify){
		case "Activity_video":{
			classify = Activity_video;
			break;
		}
		case "Nature_video":{
			classify = Nature_video;
			break;
		}
		case "Street_video":{
			classify = Street_video;
			break;
		}
		case "Rocket_video":{
			classify = Rocket_video;
			break;
		}
		case "New_video":{						//该功能暂时未上线
			classify = New_video;
			break;
		}
	}
	console.log(classify);
	var isNeed = false;			//是否需要翻页
	var num = 0;
	var j;
	for(var i in classify.totalVideo){
		if(classify.totalVideo[i].isShow==true&&isNeed==false){		//找到第一个isShow为true且判断是否需要翻页
			if(classify.totalVideo[parseInt(i)-1].isShow==false)			
			{
				j = parseInt(i) - 1;
				isNeed = true;
			}
		}
		if(isNeed==true&&num<8)
		{
			classify.totalVideo[i].isShow = false;
			num++;
		}
		if(num>=8){
			break;
		}
	}
	num = 8;
	for(;num>0;j--,num--){
		classify.totalVideo[j].isShow = true;
	}
}


function isLogin(){
	$.ajax({										//判断登录
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
                login_right_top.login = true;
            }    
        }
    });
}



function getVideo(){
	var college = getQueryString("college");
	// console.log(college);
	var collegeAllName = ["航空宇航学院","能源与动力学院","自动化学院","电子信息工程学院","机电学院","材料科学与技术学院","民航(飞行)学院","理学院","经济与管理学院","人文与社会科学学院","艺术学院","外国语学院"];


	if(college<=12){
		var collegeName = collegeAllName[college-1];
	}
	else if(college==15){
		var collegeName = "航天学院";
	}
	else if(college==16){
		var collegeName = "计算机科学与技术学院";
	}	
	else if(college==19){
		var collegeName = "国际教育学院";
	}
	collegeTitle.name = collegeName;


	var collegeData = {"college":college};
	$.ajax({									//获取视频数据
        url: "?m=video&c=index&a=AJgetVideoByCollege",
    	type: "POST",
        data: collegeData,
        error: function(request) {
            // console.log(request);
            console.log('error');
        },
        success: function(data) {
            console.log(data); 
            data = JSON.parse(data);
            console.log(data); 
            if(data){ 
            	sortData(data);
            }
        }
    });
}
//sortData(json)			将数据分类渲染
function sortData(data) {
	var activityNum = 0;
	var activityVideo = Array();
	var natureNum = 0;
	var natureVideo = Array();
	var streetNum = 0;
	var streetVideo = Array();
	var rocketNum = 0;
	var rocketVideo = Array();
	for(var i in data){
		if(data[i].classify==0){		//Activity_video
			activityVideo[activityNum] = data[i];
			activityVideo[activityNum].picSrc = "./Uploads/photo/video/" + data[i].vid + ".jpg";
			activityVideo[activityNum].videoSrc = "?m=video&c=Index&a=play&vid="+ data[i].classify + "_" + data[i].vid;
			if(activityNum<8){
				activityVideo[activityNum].isShow = true;
			}
			else{
				activityVideo[activityNum].isShow = false;
			}
			activityNum++;
			$('#Activity_box').removeClass("hide").addClass("show");
		}
		else if(data[i].classify==1){
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
			$('#Nature_box').removeClass("hide").addClass("show");			
		}
		else if(data[i].classify==2){
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
			$('#Street_box').removeClass("hide").addClass("show");			
		}
		else if(data[i].classify==3){
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
			$('#Rocket_box').removeClass("hide").addClass("show");			
		}
	}
	Activity_video.totalVideo = activityVideo;
	Nature_video.totalVideo = natureVideo;
	Street_video.totalVideo = streetVideo;
	Rocket_video.totalVideo = rocketVideo;
	console.log(Activity_video);
	console.log(Nature_video);
	console.log(Street_video);
	console.log(Rocket_video);	
}



//未完成
//1.已经实现了一行只放8个   其他的隐藏了     现在要为按钮设立点击事件     点击后换一页显示
//2.可以是下一页   也可以是更多




