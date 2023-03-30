// function upload(){
//     var imgcanvas = document.getElementById("canv1");
//     var fileinput = document.getElementById("file-browser");
//     var image = new SimpleImage(fileinput);
//     image.drawTo(imgcanvas);
//   }

const image_input = document.querySelector('#image_input')
var uploaded_image = "";
image_input.addEventListener("change",function(){
    const reader = new FileReader()
    reader.addEventListener("load", ()=> {
        uploaded_image = reader.result;
        document.querySelector('display_image').style.backgroundImage = `url(${uploaded_image})`


    });
    reader.readAsDataURL(this.files[0]);
})
console.log(uploaded_image)