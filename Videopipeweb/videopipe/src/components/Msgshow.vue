<template>
    <div v-if="show" class="mask">
      <div class="dlg-show">
        <div class="one">
          <div class="one_one">{{caption}}</div>
          <div class="one_two" @click="close">
            <el-icon><CloseBold /></el-icon>
          </div>
        </div>
        <div class="msg">{{msg}}</div>
        <div class="button">
          <el-button type="primary" class="msgconfirm" @click="confirmClick">确定</el-button>
        </div>
      </div>
    </div>
  </template>
   
  <script>
  
  export default {
      name: 'MsgShow',//组件递归必须有name属性 不然无法递归    
      props: {
          caption:{},
          show: {},
          msg:{},
      },
      data() {
          return {
          }
      },
  
      methods: {
          close() {
              this.$emit('close');
          },
          confirmClick() {
              this.$emit('close');
          }
      }
  }
  </script>
  
  <style>
      .dlg-show {
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
      .msg{
        margin-top: 20px;
        margin-bottom: 20px;
        size: 10px;
      }
      .msgconfirm{
        width: 100%;
      }
  </style>