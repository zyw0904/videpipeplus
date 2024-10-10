<template>
    <div v-if="show" class="mask">
      <div class="dlg-input-box">
        <div class="one">
          <div class="one_one">{{caption}}</div>
          <div class="one_two" @click="close">
            <el-icon><CloseBold /></el-icon>
          </div>
        </div>
        <div>
            <input class="input-box" placeholder="filename" v-model="inputValue" ref="getfocus"/>
        </div>
        <div class="button">
          <el-button class="cancel" @click="cancelClick">cancel</el-button>
          <el-button type="primary" class="confirm" @click="confirmClick">confirm</el-button>
        </div>
      </div>
      <msg-show caption="Prompting" :msg="msgText" :show="showMessage" @close="showMessage=false"></msg-show>    
    </div>
</template>
  
<script>

import MsgShow from './Msgshow.vue'
import axios from 'axios'; 

export default {
    name: 'InputBox',//组件递归必须有name属性 不然无法递归    
    props: {
        caption:{},
        value:{},
        show: {},
        devices: {},
    },
    components: {
      MsgShow
    },
    watch: {
      show(val){
          if (val == true) {
              this.$nextTick(() => {
                  this.$refs.getfocus.focus();
              })
          }
      },
      value(val){
        this.inputValue = val;
      },

    },
    data() {
        return {
          showMessage:false,
          inputValue:'',
          msgText:'',
        }
    },
    
    methods: {
      
        close() {
          this.$emit('close');
        },
        async confirmClick() {
          if (this.inputValue == "") {
              this.msgText = "Content is empty!";
              this.showMessage = true;
          } else {
              this.$emit('confirm', this.inputValue)
          } 
          try {
            const deviceIp = this.devices === 'raspberryPi' ? '192.168.137.226' : '192.168.137.226';

            // 使用获取的数据发送请求到相应设备以创建文件
            const response = await axios.post(`http://${deviceIp}:3000/api/create-file`, {
              targetFilename: this.inputValue + '.cpp',  // 现有文件名
              sourceFilename: 'Template.txt',  // 新文件名
            });
            console.log(response.data); // 打印返回的数据以供调试
          } catch (error) {
            console.error('Error creating file:', error);
          }
          this.inputValue = ""
        },
        cancelClick() {           
          this.$emit('cancel');
        }

    }
}
</script>

<style>
  .one{
    display: grid;
    grid-template-columns: 1fr 1fr;
  }
  .one_one{
    font-size: 15px;
    font-weight: 500;
    text-align: left;
    
  }
  .one_two{
    display:flex;
    justify-content: flex-end;
    margin-top: 5px;
    cursor: pointer;
  }
  .input-box{
    width: 100%;
    height: 30px;
    border-radius: 5px;
    border-color:rgb(62, 180, 248);
    margin-top: 10px;
    margin-bottom: 10px;
  }
  .dlg-input-box {
      display: block;
      border-radius: 5px;
      width: 350px;
      height: 160px;
      background-color: #fff;
      padding: 30px;
      position: absolute;
      top: 0;
      bottom: 0;
      left: 0;
      right: 0;
      margin: auto;
  }
  .button{
    display: flex;
    justify-content: flex-end;
  }
</style>