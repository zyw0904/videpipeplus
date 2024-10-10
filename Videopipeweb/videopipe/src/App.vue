<script setup>
  import { DndProvider } from 'vue3-dnd'
  import { HTML5Backend } from 'react-dnd-html5-backend'
</script>

<template>
  <DndProvider :backend="HTML5Backend">
    <router-view></router-view>
  </DndProvider>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
/* #background{
  background:url("../../public/background.jpg");
  width:100%;	
  height: 100%;		
  position:relative;
  background-size:100% 100%;
} */
/* 你的样式 */
</style>