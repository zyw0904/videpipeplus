<template>
    <div class="custom-node">
      <div class="title">
        <el-icon style="font-size: 30px;"><component :is="props.data.icon" /></el-icon>
        <span style="font-size: 25px;">{{ props.data.title }}</span>
      </div>
      <div class="input">
        <span>node_name:</span>
        <el-input class="nodrag" v-model="inputValue" placeholder="Enter text here"></el-input>
      </div>
      <div class="button">
        <el-button type="primary" @click="add_node_information">load</el-button>
      </div>
    </div>
    <div>
        <Handle :id="props.data.title" type="source" :position="Position.Right" />
        <Handle :id="props.data.title" type="target" :position="Position.Left" />
    </div>
  </template>
  
  <script setup lang="ts">
  import { Handle, Position, useVueFlow } from '@vue-flow/core'
  import { ref, defineProps } from 'vue';
  import { ElInput, ElIcon } from 'element-plus';
  import axios from 'axios';
  
  // Define props
  const props = defineProps<{
  data: {
    inputValue: string;
    device: string;
    label: string;
    title: string;
    icon: string;
  };
  }>();

  // Define state
  const inputValue = ref('');

  async function add_node_information(){
    console.log(props.data.title)
    try {
      const deviceIp = props.data.device === 'raspberryPi' ? '192.168.137.226' : '192.168.137.226';

      // 使用获取的数据发送请求到相应设备以创建文件
      const response = await axios.post(`http://${deviceIp}:3000/api/create_OSDnode`, {
        targetFilename: props.data.inputValue + '.cpp',  // 文件名
        title: props.data.title,
        icon: props.data.icon,
        node_name: inputValue.value,
      });
      console.log(response.data); // 打印返回的数据以供调试
    } catch (error) {
      console.error('Error creating file:', error);
    }
  }
  </script>
  
  <style scoped>
    .vue-flow__handle {
        height:24px;
        width:8px;
        border-radius:4px
    }

  .custom-node {
    display: grid;
    grid-template-rows: 50px 1fr;
    gap: 5px;
    border: 1px solid #ccc;
    padding: 10px;
    border-radius: 5px;
    background-color: #f9f9f9;
  }
  
  .title {
    display: flex;
    align-items: center;
    gap: 10px;
    background-color: #dfa1f8;
    padding: 5px;
    border-radius: 3px;
  }
  
  .input {
    display: flex;
    background-color: #ffffff;
    padding: 5px;
    border-radius: 3px;
  }
  </style>