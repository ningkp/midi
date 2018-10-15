

$(function(){

// 导航条的link点击事件
$('#nav li').click(function(event) {
	 // alert($('#nav li').index(this));//获取被点击元素的标签  终于找到你  还好我没放弃
	 var i = $('#nav li').index(this);
	 if(i<4) {
	 	setPage(i);
	 }
	 else
	 {}
});

//toolbar的点击事件
$('.toolbar a').click(function(event) {
	/* Act on the event */
	var i = $('.toolbar a').index(this);
	setPage(i);
});


//导航条的鼠标hover			终于搞定了
$('#top-right li').mouseover(function(event) {
	/* Act on the event */
	var i = $('#top-right li').index(this);
	setList(i);
	$('#top-right li').mouseout(function(event){
		if(event.clientY>=60){					//yes！！！漂亮
			if(i==0)
			{
				id = '#tool';
			}
			else if(i==1)
			{
				id = '#tasks';
			}
			else if(i==2)
			{
				id = '#panel1';
			}
			else if(i==3)
			{
				id = '#panel2';
			}
			$(id).mouseover(function(event) {
				/* Act on the event */
				setList(i);
				$(id).mouseout(function(event) {
					/* Act on the event */
					hideList(i);
				});
			});
			// $('#tool').mouseover(function(event) {
			// 	/* Act on the event */
			// 	setList(0);
			// 	$('#tool').mouseout(function(event) {
			// 		/* Act on the event */
			// 		hideList(0);
			// 	});
			// });
			// $('#tasks').mouseover(function(event) {
			// 	/* Act on the event */
			// 	setList(1);
			// 	$('#tasks').mouseout(function(event) {
			// 		/* Act on the event */
			// 		hideList(1);
			// 	});
			// });
			// $('#panel1').mouseover(function(event) {
			// 	/* Act on the event */
			// 	setList(2);
			// 	$('#panel1').mouseout(function(event) {
			// 		/* Act on the event */
			// 		hideList(2);
			// 	});
			// });
			// $('#panel2').mouseover(function(event) {
			// 	/* Act on the event */
			// 	setList(3);
			// 	$('#panel2').mouseout(function(event) {
			// 		/* Act on the event */
			// 		hideList(3);
			// 	});
			// });
		}
		else{
			hideList(i);
		}
	});
});

//导航条panel2的鼠标点击事件
$('.panel .panel-body a').click(function(event) {
	/* Act on the event */
	$(this).addClass('active').siblings().removeClass('active');
});


//home-page里的点击事件
$('.tool-page .home-page .my-change .nav li').click(function(event) {
	/* Act on the event */
	var i = $('.tool-page .home-page .my-change .nav li').index(this);
	setNav(i,'home');
});

//项目page里的点击事件
$('.tool-page .project-page .project-list .nav li').click(function(event) {
	/* Act on the event */
	var i = $('.tool-page .project-page .project-list .nav li').index(this);
	setNav(i,'project');
});

//任务page里的点击事件
$('.tool-page .all-page .all-list .nav li').click(function(event) {
	/* Act on the event */
	var i = $('.tool-page .all-page .all-list .nav li').index(this);
	setNav(i,'all');
});

// 好友page里的点击事件
$('.tool-page .goodFriends-page .goodFriends-list .nav li').click(function(event) {
	/* Act on the event */
	var i = $('.tool-page .goodFriends-page .goodFriends-list .nav li').index(this);
	setNav(i,'goodFriends');
});

//通知page里的点击事件
$('.tool-page .comment-page .comment-list .nav li').click(function(event) {
	/* Act on the event */
	var i = $('.tool-page .comment-page .comment-list .nav li').index(this);
	setNav(i,'comment');
});

//私信page里的点击事件
$('.tool-page .envelope-page .envelope-list .nav li').click(function(event) {
	/* Act on the event */
	var i = $('.tool-page .envelope-page .envelope-list .nav li').index(this);
	setNav(i,'envelope');
});

//账户page里的点击事件
$('.tool-page .tree-page .tree-list .nav li').click(function(event) {
	/* Act on the event */
	var i = $('.tool-page .tree-page .tree-list .nav li').index(this);
	setNav(i,'tree');
});
});


// 设置左侧工具栏的 页面的
function setPage(i) {				
	// body...
	// 设置左侧工具栏
	$('.toolbar a:eq(' + i + ')').addClass('active').siblings().removeClass();

	// 设置页面
	id = '#page' + (i+1);			
	$(id).addClass("show").removeClass('hide').siblings().addClass('hide').removeClass('show');		//perfect!
}

//设置home-page下的nav
function setNav(i,type){
	if(type=='home'){
		$('.tool-page .home-page .my-change .nav li:eq(' + i + ')').addClass('active').siblings().removeClass();		
	}
	else{
		$('.tool-page .' + type + '-page .' + type + '-list .nav li:eq(' + i + ') a').addClass('active').siblings().removeClass();
	 	$('.tool-page .' + type + '-page .' + type + '-list .nav li:eq(' + i + ')').siblings().find('a').removeClass('active');
	 	console.log(type);
	}
}



//设置list
function setList(i){
	if(i==0)
	{
		$('#tool').addClass('show');
	}
	else if(i==1)
	{
		$('#tasks').addClass('show');
	}
	else if(i==2)
	{
		$('#panel1').addClass('show');
	}
	else if(i==3)
	{
		$('#panel2').addClass('show');
	}
}
// 隐藏list
function hideList(i){
	if(i==0)
	{
		$('#tool').removeClass('show');
	}
	else if(i==1)
	{
		$('#tasks').removeClass('show');
	}
	else if(i==2)
	{
		$('#panel1').removeClass('show');
	}
	else if(i==3)
	{
		$('#panel2').removeClass('show');
	}
}