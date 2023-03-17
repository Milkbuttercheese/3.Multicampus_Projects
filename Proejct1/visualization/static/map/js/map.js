$(function (){
    $('#mymap > div').css({'width': '100%', 'height':'100%'});
    $('#mymap > div > div').css({'width': '100%', 'height':'100%', 'padding-bottom': '0'});

});

onload=function(){
            navigator.geolocation.getCurrentPosition(showYourLocation, showErrorMsg, options);

            function showYourLocation(position) {  // 성공했을때 실행
                var userLat = position.coords.latitude;
                var userLng = position.coords.longitude;

                $('#userLat').val(userLat)
                $('#userLng').val(userLng)

                console.log(userLat)
                console.log(userLng)
                console.log(typeof(userLat))
            }

            function showErrorMsg(error) { // 실패했을때 실행
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                    loc.innerHTML = "사용자가 Geolocation API의 사용 요청이 거부되었습니다."
                    break;

                    case error.POSITION_UNAVAILABLE:
                    loc.innerHTML = "가져온 위치 정보를 사용할 수 없습니다."
                    break;

                    case error.TIMEOUT:
                    loc.innerHTML = "위치 정보를 가져오기 위한 요청이 허용 시간이 초과되었습니다."
                    break;

                    case error.UNKNOWN_ERROR:
                    loc.innerHTML = "알 수 없는 오류가 발생"
                    break;
                }
            }

            function options(options){
                var options = {enableHighAccuracy: True}
            }

        }