function setCookie(cname, cvalue, exdays) {
    var d = new Date();
    d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
    var expires = "expires=" + d.toUTCString();
    document.cookie = cname + "=" + cvalue + "; " + expires;
}
//获取cookie
function getCookie(cname) {
    var name = cname + "=";
    var ca = document.cookie.split(';');
    for (var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') c = c.substring(1);
        if (c.indexOf(name) != -1) return c.substring(name.length, c.length);
    }
    return "";
}
//清除cookie  
function clearCookie(name) {
    setCookie(name, "", -1);
}
//查询cookie
function checkCookie(name) {
    var _name = getCookie("name");
    if (true) {}
}



function loginOut() {

    $.ajax({
        type: "POST",
        url: "?m=user&c=login&a=logout",
        data: {
            action: 'logout',
        },
        success: function(data) {
            console.log(data);
            clearCookie('uid');
            location.reload();
        },
        error: function(request) {
            alert("Connection error");
        },
    });
}

function is_Login() {
    var _uid = getCookie("uid");
    return _uid == "" ? false : true;


}

function collection(vid) {
    $.ajax({
        type: "POST",
        url: "?m=video&c=comment&a=collection",
        data: {
            action: 'collection',
            vid:vid,
        },
        success: function(data) {
            console.log(data);
            
            // clearCookie('uid');
        },
        error: function(request) {
            alert("Connection error");
        },
    });
}
//
//
//jq扩展，读取url参数
(function ($) {
    $.getUrlParam = function (name) {
        var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)");
        var r = window.location.search.substr(1).match(reg);
        if (r != null) return unescape(r[2]); return null;
    }
})(jQuery);
