<<<<<<< HEAD
$(document).ready(function(){
    for (var i=1; i < 6; i++){
        $('#pic' + i).click(function(e){
            alert(this.id);
=======
$(window).on('load', function() {
    for (var i = 1; i < 6; i++) {
        $('#choose' + i).click(function() {
            alert("선택" + i);
>>>>>>> 8ba5c7257d8c05b5e9726c665bf5fc00c5990127
        });
    }
});