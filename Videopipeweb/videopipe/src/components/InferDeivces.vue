<script setup>
//import HelloWorld from './components/InferDeivces.vue'
// import TheWelcome from './components/TheWelcome.vue'
</script>

<script>
import axios from 'axios';

export default {
  name: 'App',
  methods: {
    async runRaspberryPiScript() {
      try {
        console.log('Sending request to Raspberry Pi...');
        const response = await fetch('http://192.168.137.226:3000/connect', {
          method: 'POST',
        });
        const data = await response.json();
        console.log('Response received:', data);
      } catch (error) {
        console.error('Error running script:', error);
      }
    },
    async handleClick(device) {
      this.$router.push({ path: '/another', query:{ device }});
      try {
        const deviceIp = device === 'raspberryPi' ? '192.168.137.226' : '192.168.137.226';
        // 发送请求到指定设备并获取数据
        const response = await axios.post(`http://${deviceIp}:3000/connect`, {
          method: 'POST',
        });
        console.log(response.data);

      } catch (error) {
        console.error('Error connecting to backend:', error);
      }
    }
  }
}
</script>

<template>
  <!-- <header>
    <img alt="Vue logo" class="logo" src="./assets/logo.svg" width="125" height="125" />

    <div id="wrapper">
      <HelloWorld msg="You did it!" />
    </div>
  </header> -->
  <div>
    <div class="grid-rows">
      <img alt="Vue logo" class="logo" src="../../public/logo.png" width="200" height="180" />
      <div>
        <h1 class="green">
          VideoPipePlus
        </h1>
        <h2>
          Select a device to start the video analysis task:
        </h2>
      </div>
      <div class="grid-columns">
        <div class = "border_1" @click="handleClick('jetsonTx2')">
          <img alt="logo_1" class="tx2" src="../../public/R-C.jpg" width="250" height="180" />
          <div>
            <h3>
              Video analysis on the GPU
            </h3>
          </div>
        </div>
        <div class = "border_2" @click="handleClick('raspberryPi')">
          <img alt="logo_2" class="coral" src="../../public/coral.jpg" width="250" height="150" />
          <div>
            <h4>
              Video analysis on the TPU
            </h4>
          </div>
        </div>
        <div class = "border_3" @click="handleClick('raspberryPi')">
          <img alt="logo_3" class="ncs2" src="../../public/intel_ncs2.jpg" width="250" height="150" />
          <div>
            <h5>
              Video analysis on the VPU
            </h5>
          </div>
        </div>
      </div>
    </div>
  </div>
  <!-- <main>
    <TheWelcome />
  </main> -->
</template>

<style scoped>
.tx2{
  position: relative;
  top: 50px;
}
.coral{
  position: relative;
  top: 50px;
}
.ncs2{
  position: relative;
  top: 50px;
}
h1 {
  text-align:center;
  font-weight: 1000;
  font-size: 2.6rem;
  position: relative;
  top: -50px;
}
h2 {
  text-align:center;
  font-weight: 500;
  font-size: 2.0rem;
  position: relative;
  top: -30px
}
h3{
  text-align:center;
  font-weight: 500;
  font-size: 1.5rem;
  position: relative;
  top: 100px
}
h4{
  text-align:center;
  font-weight: 500;
  font-size: 1.5rem;
  position: relative;
  top: 130px
}
h5{
  text-align:center;
  font-weight: 500;
  font-size: 1.5rem;
  position: relative;
  top: 130px
}
.border_1{
  height: 400px;
  width: 300px;
  border: 1px solid var(--el-border-color);
  border-radius: 20px;
  text-align:center;
  margin: 0 auto;
  box-shadow: var(--el-box-shadow);
  background-color: white;
}
.border_2{
  height: 400px;
  width: 300px;
  border: 1px solid var(--el-border-color);
  border-radius: 20px;
  text-align:center;
  margin: 0 auto;
  box-shadow: var(--el-box-shadow);
  background-color: white;
}
.border_3{
  height: 400px;
  width: 300px;
  border: 1px solid var(--el-border-color);
  border-radius: 20px;
  text-align:center;
  margin: 0 auto;
  box-shadow: var(--el-box-shadow);
  background-color: white;
}
.border_1:hover{
  box-shadow: var(--el-box-shadow-dark);
}
.border_2:hover{
  box-shadow: var(--el-box-shadow-dark);
}
.border_3:hover{
  box-shadow: var(--el-box-shadow-dark);
}
.logo {
  display: block;
  margin: 0 auto 2rem;
}

/* @media (min-width: 1024px) {
  header {
    display: block;
    place-items: center;
    padding-right: calc(var(--section-gap) / 2);
  }

  .logo {
    margin: 0 2rem 0 0;
  }

  header .wrapper {
    display: flex;
    place-items: flex-start;
    flex-wrap: wrap;
  }
} */
</style>
