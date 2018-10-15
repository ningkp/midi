 //$(function () { $('.form-control').tooltip('toggle');});

    // 用户名验证
    $('#user_name').focus(function() {
        $(this).on('input', function() {
            if($.isNumeric($(this).val().length==0)) {  // 判断第一个字符是不是数字
                $(this).parents('.form-group').addClass('has-error');
                $(function () { $('#user_name').tooltip('toggle');});
            } else {
                $(this).parents('.form-group').removeClass('has-error');
            }
        });
    }).blur(function() {
        if($(this).val().length == 0) {
            $(this).parents('.form-group').addClass('has-error');            
                $(function () { $('#user_name').tooltip('toggle');});
        }else{
        			//在这里验证用户名是否已存在--------------
              	if(!($(this).val().length <= 20&&$(this).val().length >= 3)){

              		$(function () { $('#user_name').tooltip('toggle');});
              	}else{

                $(function () { $('#user_name').tooltip('destroy');});
              	}
        }
    });


   // 邮箱
    $('#e_mail').focus(function() {
        $(this).on('input', function() {
            if($(this).val().length == 0) {
                $(this).parents('.form-group').addClass('has-error');
                $(function () { $('#e_mail').tooltip('toggle');});
            } else {
                $(this).parents('.form-group').removeClass('has-error');
            }
        });
    }).blur(function() {
        if(!$(this).val().match(/^\w+((-\w+)|(\.\w+))*\@[A-Za-z0-9]+((\.|-)[A-Za-z0-9]+)*\.[A-Za-z0-9]+$/)) {
            $(this).parents('.form-group').addClass('has-error');
            
             $(function(){$('#e_mail').tooltip("toggle");});
        }
        else{

            $(function(){$('#e_mail').tooltip("destroy");});
           

        }
    });

   // 密码验证
    $('#password').focus(function() {
        $(this).on('input', function() {
            if($(this).val().length == 0) {
                $(this).parents('.form-group').addClass('has-error');
                $(function(){$('#password').tooltip("toggle");});
            } else {
                $(this).parents('.form-group').removeClass('has-error');
            }
        });
    }).blur(function() {
        if($(this).val().length == 0) {
            $(this).parents('.form-group').addClass('has-error');
            $(function(){$('#password').tooltip("toggle");});
        }else{       	
        $(function(){$('#password').tooltip("destroy");});
        }
    });

    $('#repassword').focus(function() {
        $(this).on('input', function() {
            if($(this).val().length == 0) {
                $(this).parents('.form-group').addClass('has-error');
                $(function(){$('#repassword').tooltip("toggle");});
            } else {
                $(this).parents('.form-group').removeClass('has-error');
            }
        });
    }).blur(function() {
        if($(this).val() != $('#password1').val()) {
            $(this).parents('.form-group').addClass('has-error');
           $(function(){$('#repassword').tooltip("toggle");});
        }else{       	
        $(function(){$('#repassword').tooltip("destroy");});
              }
    });


$(document).ready(function () {
    if(($('#user_name').val().length <= 20&&$('#user_name').val().length >= 3)&&($('#e_mail').val().match(/^\w+((-\w+)|(\.\w+))*\@[A-Za-z0-9]+((\.|-)[A-Za-z0-9]+)*\.[A-Za-z0-9]+$/))&&($('#password').val() == $('#repassword').val())&&($('#password').val())){
        $('#sign_upsubmit').attr("disabled",false);
    }else{
        $('#sign_upsubmit').attr("disabled",true);
    }

});

