<script setup lang="ts">
import { h, ref, watch, onMounted } from 'vue';
import { VideoCamera, Operation, Monitor, Film, Cpu } from '@element-plus/icons-vue';
import { Search } from '@element-plus/icons-vue'
import MsgShow from './Msgshow.vue';
import InputBox from './InputBox.vue';
import { useRoute } from 'vue-router';
import { VueFlow, Position, Panel, useVueFlow, MarkerType, EdgeChange, NodeChange } from '@vue-flow/core'
import { Background } from '@vue-flow/background'
import { Controls } from '@vue-flow/controls'
import { MiniMap } from '@vue-flow/minimap'
import CustomNode_inference from './CustomNode_inference.vue';
import CustomNode_input from './CustomNode_input.vue';
import CustomNode_output from './CustomNode_output.vue';
import CustomNode_OSD from './CustomNode_OSD.vue';
import CustomNode_rtmp from './CustomNode_rtmp.vue';
import { useDialog } from './useDialog'
import Dialog from './Dialog.vue'
import axios from 'axios';

const { addEdges, onNodesChange, onEdgesChange, applyNodeChanges, applyEdgeChanges } = useVueFlow()

const dialog = useDialog()

let value1 = ref(false)
let isBorderVisible = ref(false)

async function handleSwitchChange(value) {

  // 通过 HTTP 请求将数据发送到后端
  try {
      const deviceIp = device.value === 'raspberryPi' ? '192.168.137.226' : '192.168.137.226';
      // 使用获取的数据发送请求到相应设备以创建文件
      const response = await axios.post(`http://${deviceIp}:3000/api/track`, {
        targetFilename: inputValue.value + '.cpp',
        track: value,
      } );
      console.log(response.data); // 打印返回的数据以供调试
  } catch (error) {
      console.error('Error creating file:', error);
  }
}

function dialogMsg(id) {
  return h(
    'span',
    {
      style: {
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '8px',
      },
    },
    [`Are you sure?`, h('br'), h('span', `[ELEMENT_ID: ${id}]`)],
  )
}

onNodesChange(async (changes) => {
  const nextChanges: NodeChange[] = [];
  let isConfirmed;
  for (const change of changes) {
    if (change.type === 'remove') {
      isConfirmed = await dialog.confirm(dialogMsg(change.id))

      if (isConfirmed) {
        nextChanges.push(change)
      }
    } else {
      nextChanges.push(change)
    }
  }
  applyNodeChanges(nextChanges)
  if(isConfirmed) {
    const firstChange = nextChanges[0] as { id: string, type: string };
    try {
        const deviceIp = device.value === 'raspberryPi' ? '192.168.137.226' : '192.168.137.226';
        // 使用获取的数据发送请求到相应设备以创建文件
        const response = await axios.post(`http://${deviceIp}:3000/api/delete_node`, {
            targetFilename: inputValue.value + '.cpp',
            title: firstChange.id,
        });
        console.log(response.data); // 打印返回的数据以供调试
    } catch (error) {
        console.error('Error creating file:', error);
    }
  }
})

onEdgesChange(async (changes) => {
  const nextChanges: EdgeChange[] = [];
  let isConfirmed; 
  for (const change of changes) {
    if (change.type === 'remove') {
      isConfirmed = await dialog.confirm(dialogMsg(change.id))

      if (isConfirmed) {
        nextChanges.push(change)
        console.log(nextChanges)
      }
    } else {
      nextChanges.push(change)
    }
  }
  applyEdgeChanges(nextChanges)
  if(isConfirmed) {
    const firstChange = nextChanges[0] as { id: string, source: string, target: string, sourceHandle: string, targetHandle: string, type: string };
    try {
        const deviceIp = device.value === 'raspberryPi' ? '192.168.137.226' : '192.168.137.226';
        // 使用获取的数据发送请求到相应设备以创建文件
        const response = await axios.post(`http://${deviceIp}:3000/api/delete_edge`, {
            targetFilename: inputValue.value + '.cpp',
            sourceHandle: firstChange.sourceHandle,
            targetHandle: firstChange.targetHandle,
        });
        console.log(response.data); // 打印返回的数据以供调试
    } catch (error) {
        console.error('Error creating file:', error);
    }
  }
})

// 获取路由参数
const route = useRoute();
const device = ref('');
onMounted(() => {
    const queryDevice = route.query.device;

    // 确保 device 是 string 类型
    if (typeof queryDevice === 'string') {
        device.value = queryDevice;
    }
});
// 定义响应式数据
const inputValue = ref('');
const caption = ref('');
const msgText = ref('');
const showMsgShow = ref(false);
const showInput = ref(false);
const dragValue = ref(''); // 新增响应式引用以存储 MsgShowClick 的 value
function InputBoxClick() {
    caption.value = "Create inference file";
    inputValue.value = "";
    showInput.value = true;
}

async function CompileClick() {
    try {
        const deviceIp = device.value === 'raspberryPi' ? '192.168.137.226' : '192.168.137.226';
        // 使用获取的数据发送请求到相应设备以创建文件
        const response = await axios.post(`http://${deviceIp}:3000/api/compile`, {
            devices: device.value,
        });
        console.log(response.data); // 打印返回的数据以供调试
        displayOutput(response.data);
    } catch (error) {
        console.error('Error creating file:', error);
    }
}

async function RunClick() {
    try {
        const deviceIp = device.value === 'raspberryPi' ? '192.168.137.226' : '192.168.137.226';
        // 使用获取的数据发送请求到相应设备以创建文件
        const response = await axios.post(`http://${deviceIp}:3000/api/run`, {
            targetFilename: inputValue.value,
        });
        console.log(response.data); // 打印返回的数据以供调试
        displayOutput(response.data);
    } catch (error) {
        console.error('Error creating file:', error);
    }
}

function displayOutput(data) {
    const outputDiv = document.getElementById('output'); // 你页面中显示输出的 div
    outputDiv!.textContent += data + '\n';  // 将新输出追加到 div 中
}

let flowKey = 0

function MsgShowClick(value) {
    showInput.value = false;
    caption.value = "Prompting";
    msgText.value = "Complete operation!";
    inputValue.value = value;
    dragValue.value = value; // 将值存储到 dragValue 中
    showMsgShow.value = true;
    isBorderVisible.value = true;
    flowKey += 1;
}

let id = 0

/**
 * @returns {string} - A unique id.
 */
function getId() {
  return `dndnode_${id++}`
}

/**
 * In a real world scenario you'd want to avoid creating refs in a global scope like this as they might not be cleaned up properly.
 * @type {{draggedType: Ref<string|null>, isDragOver: Ref<boolean>, isDragging: Ref<boolean>}}
 */
const state = {
  /**
   * The type of the node being dragged.
   */
  draggedType: ref(null),
  isDragOver: ref(false),
  isDragging: ref(false),
}

const { draggedType, isDragOver, isDragging } = state

const nodeTypes = {
  custom_inference: CustomNode_inference,
  custom_input: CustomNode_input,
  custom_output: CustomNode_output,
  custom_OSD: CustomNode_OSD,
  custom_rtmp: CustomNode_rtmp,
  
};

const { addNodes, screenToFlowCoordinate, onNodesInitialized, updateNode } = useVueFlow()

watch(isDragging, (dragging) => {
document.body.style.userSelect = dragging ? 'none' : ''
})

const storedData = ref(); // 使用 ref 存储数据

function onDragStart(event, type, item) {
    if (event.dataTransfer) {
    const data = JSON.stringify({
      type: type,
      icon: item.icon,
      text: item.label,
    });
    storedData.value = data;
    event.dataTransfer.setData('application/vueflow', data)
    event.dataTransfer.effectAllowed = 'move'
    }

    draggedType.value = type
    isDragging.value = true

    document.addEventListener('drop', onDragEnd)
}

  /**
   * Handles the drag over event.
   *
   * @param {DragEvent} event
   */
  function onDragOver(event) {

    const { type, icon, text } = JSON.parse(storedData.value);
    event.preventDefault()
    if (draggedType.value) {
      isDragOver.value = true

      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = 'move'
      }
    }
  }

  function onDragLeave() {
    isDragOver.value = false
  }

  function onDragEnd() {
    isDragging.value = false
    isDragOver.value = false
    draggedType.value = null
    document.removeEventListener('drop', onDragEnd)
  }

  /**
   * Handles the drop event.
   *
   * @param {DragEvent} event
   */
   async function onDrop(event) {
    const position = screenToFlowCoordinate({
      x: event.clientX,
      y: event.clientY,
    })

    const dataStr = event.dataTransfer.getData('application/vueflow');
    const { type, icon, text } = JSON.parse(dataStr);

    const nodeId = getId()

    let newNodeType;

    if (icon === 'VideoCamera') {
    newNodeType = 'custom_input';
    } else if (icon === 'Operation') {
    newNodeType = 'custom_inference';
    } else if (icon === 'Monitor') {
    newNodeType = 'custom_OSD';
    } else if (icon === 'Film') {
    newNodeType = 'custom_output';
    } else if (icon === 'Cpu') {
    newNodeType = 'custom_rtmp';
    } else {
    newNodeType = 'default';
    }

    const newNode = {
      id: text,
      type: newNodeType,
      position,
    //   targetPosition: Position.Left,
    //   sourcePosition: Position.Right,
      data: { inputValue: inputValue,
              device: device,
              label: nodeId,
              title: text,
              icon: icon,
      },
    }

    /**
     * Align node position after drop, so it's centered to the mouse
     *
     * We can hook into events even in a callback, and we can remove the event listener after it's been called.
     */
    const { off } = onNodesInitialized(() => {
      updateNode(nodeId, (node) => ({
        position: { x: node.position.x - node.dimensions.width / 2, y: node.position.y - node.dimensions.height / 2 },
      }))

      off()
    })

    addNodes(newNode)
}

const nodes = ref([])

async function onConnect(params) {
  console.log('on connect', params.sourceHandle)
  console.log('on connect', params.targetHandle)
  addEdges(params)
  try {
    const deviceIp = device.value === 'raspberryPi' ? '192.168.137.226' : '192.168.137.226';
    // 使用获取的数据发送请求到相应设备以创建文件
    const response = await axios.post(`http://${deviceIp}:3000/api/node_link`, {
        targetFilename: inputValue.value + '.cpp',
        source: params.sourceHandle,
        target: params.targetHandle,
    });
    console.log(response.data); // 打印返回的数据以供调试
  } catch (error) {
    console.error('Error creating file:', error);
  }
}

const input = ref('');
const menus = ref({
    '1': {
    icon: "VideoCamera",
    title: 'Input Node',
    items: [
        { index: '1-1', label: 'vp_src_node', icon: "VideoCamera" },
    ],
    },
    '2': {
    icon: "Operation",
    title: 'Inference Node',
    items: [
        { index: '2-1', label: 'vp_fire_detect_node_tpu', icon: "Operation" },
        { index: '2-2', label: 'vp_general_infer_node', icon: "Operation" },
        { index: '2-3', label: 'vp_mobilenet_edgetpu_face_node', icon: "Operation" },
        { index: '2-4', label: 'vp_mobilenet_edgetpu_object_node', icon: "Operation" },
        { index: '2-5', label: 'vp_plate_detect_node_tpu', icon: "Operation" },
        { index: '2-6', label: 'vp_yunnet_face_detector', icon: "Operation" },
        { index: '2-7', label: 'vp_sface_feature_encoder_node', icon: "Operation" },
    ],
    },
    '3': {
    icon: "Monitor",
    title: 'OSD Node',
    items: [
        { index: '3-1', label: 'vp_face_osd_node', icon: "Monitor" },
        { index: '3-2', label: 'vp_face_osd_node_v2', icon: "Monitor" },
        { index: '3-3', label: 'vp_osd_node', icon: "Monitor" },
        { index: '3-4', label: 'vp_osd_node_v2', icon: "Monitor" },
    ],
    },
    '4': {
    icon: "Film",
    title: 'Output Node',
    items: [
        { index: '4-1', label: 'vp_screen_des_node',icon: "Film" },
    ],
    },
    '5': {
    icon: "Cpu",
    title: 'Rtmp Node',
    items: [
        { index: '5-1', label: 'vp_rtmp_des_node', icon: "Cpu" },

    ],
    },
});

const filteredMenuItems = (menuIndex) => {
    if (!input.value) {
    return menus.value[menuIndex].items;
    }
    return menus.value[menuIndex].items.filter(item =>
    item.label.toLowerCase().includes(input.value.toLowerCase())
    );
};

const handleOpen = (key, keyPath) => {
    console.log('open:', key, keyPath);
};

const handleClose = (key, keyPath) => {
    console.log('close:', key, keyPath);
};

</script>

<template>
    <div class="rows">
        <div class="border_4">
            <el-tooltip content="Create infer file">
                <el-icon class="DocumentAdd" @click="InputBoxClick">
                    <DocumentAdd />
                </el-icon>
            </el-tooltip>
            <el-tooltip content="Compile infer file">
                <el-icon class="Setting" @click="CompileClick">
                    <Setting />
                </el-icon>
            </el-tooltip>
            <el-tooltip content="Run infer file">
                <el-icon class="CaretRight" @click="RunClick">
                    <CaretRight />
                </el-icon>
            </el-tooltip>
        </div>
        <div class="columns">
            <div class="border_5">
                <div>
                    <span class="search">Search:</span>
                    <el-input v-model="input" class="searchbox" style="width: 240px;height: 30px;" :prefix-icon="Search" clearable>
                    </el-input>
                </div>
                
                <el-scrollbar height="520px">
                    <el-row class="tac">
                        <el-col :span="40">
                        <el-menu
                            default-active="2"
                            background-color="rgb(231, 230, 230)"
                            text-color="black"
                            class="el-menu-vertical-demo"
                            @open="handleOpen"
                            @close="handleClose"
                        >
                            <el-sub-menu index="1" v-if="filteredMenuItems('1').length > 0">
                                <template #title>
                                    <el-icon style="font-size: 20px;"><VideoCamera /></el-icon>
                                    <span class="myfont">Input Node</span>
                                </template>
                                <el-menu-item
                                    class="source-box"
                                    v-for="item in filteredMenuItems('1')"
                                    :key="item.index"
                                    :index="item.index"
                                    style="font-size: 20px;"
                                    :draggable="true" 
                                    @dragstart="onDragStart($event, 'input', item)"
                                    >
                                    <!-- <Box :type="ItemTypes.BOX" :name="item.label" :is-dropped="isDropped(item.label)" :icon="menus[1].icon">
                                    </Box> -->
                                    <el-icon style="font-size: 20px;"><component :is="item.icon" /></el-icon>
                                    {{ item.label }}
                                </el-menu-item>
                            </el-sub-menu>
                            <el-sub-menu index="2" v-if="filteredMenuItems('2').length > 0">
                                <template #title>
                                    <el-icon style="font-size: 20px;"><Operation /></el-icon>
                                    <span class="myfont">Inference Node</span>
                                </template>
                                <el-menu-item
                                    class="source-box"
                                    v-for="item in filteredMenuItems('2')"
                                    :key="item.index"
                                    :index="item.index"
                                    style="font-size: 20px;"
                                    :draggable="true"
                                    @dragstart="onDragStart($event, 'default', item)"
                                    >
                                        <!-- <Box :type="ItemTypes.BOX" :name="item.label" :is-dropped="isDropped(item.label)" :icon="menus[2].icon">
                                        </Box> -->
                                    <el-icon style="font-size: 20px;"><component :is="item.icon" /></el-icon>
                                    {{ item.label }}
                                </el-menu-item>
                            </el-sub-menu>
                            <el-sub-menu index="3" v-if="filteredMenuItems('3').length > 0">
                                <template #title>
                                    <el-icon style="font-size: 20px;"><Monitor /></el-icon>
                                    <span class="myfont">OSD Node</span>
                                </template>
                                <el-menu-item
                                    class="source-box"
                                    v-for="item in filteredMenuItems('3')"
                                    :key="item.index"
                                    :index="item.index"
                                    style="font-size: 20px;"
                                    :draggable="true" 
                                    @dragstart="onDragStart($event, 'default', item)"
                                    >
                                    <!-- <Box :type="ItemTypes.BOX" :name="item.label" :is-dropped="isDropped(item.label)" :icon="menus[3].icon">
                                    </Box> -->
                                    <el-icon style="font-size: 20px;"><component :is="item.icon" /></el-icon>
                                    {{ item.label }}
                                </el-menu-item>
                            </el-sub-menu>
                            <el-sub-menu index="4" v-if="filteredMenuItems('4').length > 0">
                                <template #title>
                                    <el-icon style="font-size: 20px;"><Film /></el-icon>
                                    <span class="myfont">Output Node</span>
                                </template>
                                <el-menu-item
                                    class="source-box"
                                    v-for="item in filteredMenuItems('4')"
                                    :key="item.index"
                                    :index="item.index"
                                    style="font-size: 20px;"
                                    :draggable="true" 
                                    @dragstart="onDragStart($event, 'output', item)"
                                    >
                                    <!-- <Box :type="ItemTypes.BOX" :name="item.label" :is-dropped="isDropped(item.label)" :icon="menus[4].icon">
                                    </Box> -->
                                    <el-icon style="font-size: 20px;"><component :is="item.icon" /></el-icon>
                                    {{ item.label }}
                                </el-menu-item>
                            </el-sub-menu>
                            <el-sub-menu index="5" v-if="filteredMenuItems('5').length > 0">
                                <template #title>
                                    <el-icon style="font-size: 20px;"><Cpu /></el-icon>
                                    <span class="myfont">Rtmp Node</span>
                                </template>
                                <el-menu-item
                                    class="source-box"
                                    v-for="item in filteredMenuItems('5')"
                                    :key="item.index"
                                    :index="item.index"
                                    style="font-size: 20px;"
                                    :draggable="true" 
                                    @dragstart="onDragStart($event, 'output', item)"
                                    >
                                    <!-- <Box :type="ItemTypes.BOX" :name="item.label" :is-dropped="isDropped(item.label)" :icon="menus[5].icon">
                                    </Box> -->
                                    <el-icon style="font-size: 20px;"><component :is="item.icon" /></el-icon>
                                    {{ item.label }}
                                </el-menu-item>
                            </el-sub-menu>
                        </el-menu>
                        </el-col>    
                    </el-row>
                </el-scrollbar>
            </div>
            <div class="border_6">
                <div class="inputvalue">
                  <p>{{inputValue}}</p>
                </div>
                <div class="border_8" @drop="onDrop">
                    <VueFlow :key="flowKey"  :nodes="nodes" :node-types="nodeTypes as any" :apply-default="false" fit-view-on-init @connect="onConnect" @dragover="onDragOver" @dragleave="onDragLeave">
                        <Background :size="2" :gap="20" pattern-color="#BDBDBD" />
                        <Controls />
                        <MiniMap />
                        <Dialog />
                        <div v-if="isBorderVisible" class="border_11">
                          <el-switch
                            v-model="value1"
                            class="mb-2"
                            active-text="object track"
                            @change="handleSwitchChange"
                          />
                        </div>
                    </VueFlow> 
                </div> 
            </div>
        </div>
        <div class="border_7">
            <div></div>
            <div id = "output" class = "border_10">

            </div>
        </div>
    </div> 
    <MsgShow :caption="caption" :msg="msgText" :show="showMsgShow" @close="showMsgShow=false">
    </MsgShow>    
    <InputBox :caption="caption" :show="showInput" :devices = "device" :value="inputValue" @close="showInput=false" @confirm="MsgShowClick" @cancel="showInput=false">
    </InputBox>
</template>
  
<script lang="ts">
  //   import MsgShow from './Msgshow.vue' 
  //   import InputBox from './InputBox.vue' 
  //   import axios from 'axios';

  //   export default {
  //   name: 'Main',
  //   components: {
  //     InputBox, MsgShow
  //   },
  //   data () {
  //     return {
 
  //       inputValue:'',
  //       caption: '',
  //       msgText: '',       
  //       showMsgShow: false,
  //       showInput: false,
  //       device: this.$route.query.device || 'raspberryPi',
  //     }
  //   },
  //   methods: {
  //     InputBoxClick() {
  //       this.caption = "Create inference file";
  //       this.inputValue = "";
  //       this.showInput = true;
  //     },     
  //     MsgShowClick(value) {
  //       this.showInput = false;
  //       this.caption = "Prompting";
  //       this.msgText = "Complete operation!";
  //       this.inputValue = value;
  //       this.showMsgShow = true;
  //     },
  //     // inputBoxYes(value){
  //     //   console.log(value);
  //     //   this.showInput = false;
 
  //     //   this.caption = "Prompting";
  //     //   this.msgText = "The value is [" + value + "]";
  //     //   this.showMsgShow = true;
 
  //     // }
  //   },
 
    
  // }
</script>

<style scoped>

@import "@vue-flow/core/dist/style.css";
@import "@vue-flow/core/dist/theme-default.css";
@import '@vue-flow/controls/dist/style.css';
@import '@vue-flow/minimap/dist/style.css';

.source-box {
  cursor: move;
}
.myfont{
    /* font-family: 'SimSun', sans-serif; */
    font-size:23px;
    font-weight:500;
    margin-left: 5px;

}
.tac{
    display: block;
    margin: 0 auto;
    width: 100%;
}
.searchbox{
    size:small;
    width: 240px;
    position: relative;
    top:5px;
    margin-left: 15px;
}
.search{
    color:black;
    font-size: 20px;
    font-weight: 500;
    position: relative;
    top:5px;
    margin-left:15px;
}
.rows{
  display: grid;
  grid-template-rows: 40px 1fr 200px; /* 定义三行栅格 */
  gap: 5px;
}
.columns{
  display: grid;
  grid-template-columns: 400px 1fr; /* 定义两行栅格 */
  gap: 5px;
}
.DocumentAdd{
    color:black;
    font-size: 30px;
    top: 4px;
    margin: 0 auto;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}
.DocumentAdd:hover{
    /* background-color:beige; */
    box-shadow: var(--el-box-shadow-lighter);
    font-size: 28px;
}
.Setting{
    color:black;
    font-size: 30px;
    top: 4px;
    margin: 0 auto;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}
.Setting:hover{
    box-shadow: var(--el-box-shadow-lighter);
    font-size: 28px;
}
.CaretRight{
    color:black;
    font-size: 30px;
    top: 4px;
    margin: 0 auto;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}
.CaretRight:hover{
    box-shadow: var(--el-box-shadow-lighter);
    font-size: 28px;
}
.border_4{
  background-color:rgb(255, 255, 255);
  display: grid;
  grid-template-columns: 1fr 1fr 1fr; /* 定义三行栅格 */
  margin: 0 auto;
  height: 40px;
  width: 1800px;
  border: 1px solid var(--el-border-color);
  border-radius: 10px;
  text-align:left;
  box-shadow: var(--el-box-shadow-lighter);
}
.border_5{
  display: grid;
  grid-template-rows: 50px 1fr; /* 定义三行栅格 */
  gap: 5px;
  /* margin: 0 auto; */
  height: 600px;
  width: 400px;
  border: 1px solid var(--el-border-color);
  border-radius: 10px;
  text-align:left;
  box-shadow: var(--el-box-shadow-lighter);
  background-color:rgb(255, 255, 255);
}
.border_6{
  display: grid;
  grid-template-rows: 50px 1fr; /* 定义三行栅格 */
  /* margin: 0 auto; */
  height: 600px;
  width: 100%;
  border: 1px solid var(--el-border-color);
  border-radius: 10px;
  text-align:left;
  box-shadow: var(--el-box-shadow-lighter);
  background-color:rgb(255, 255, 255);
}
.border_7{
  display: grid;
  grid-template-rows: 30px 1fr; /* 定义三行栅格 */
  gap: 5px;
  /* margin: 0 auto; */
  height: 250px;
  width: 100%;
  border: 1px solid var(--el-border-color);
  border-radius: 10px;
  text-align:left;
  box-shadow: var(--el-box-shadow-lighter);
  background-color: rgb(231, 230, 230);
}
.border_8{
  flex-direction:column;
  display:flex;
  margin: 0 auto;
  height: 99%;
  width: 99%;
  border: 1px solid var(--el-border-color);
  text-align:left;
  box-shadow: var(--el-box-shadow-lighter);
  background-color:rgba(164, 169, 172, 0.575);
}
.border_10{
    word-wrap: break-word;  /* 强制长单词或 URL 换行 */
    word-break: break-all;  /* 强制长单词在任意字符点换行 */
    white-space: pre-wrap;  /* 保留空白符并允许自动换行 */
    overflow-wrap: break-word;  /* 处理长单词换行 */
    overflow-y: auto; /* 当内容溢出时显示滚动条 */
    flex-direction:column;
    display:flex;
    margin: 0 auto;
    height: 95%;
    width: 99%;
    /* border-radius: 10px; */
    background-color:rgb(255, 255, 255);
}
.border_11{
  position: absolute;
  top: 20px; /* 调整此值可改变按钮的垂直位置 */
  right: 20px; /* 调整此值可改变按钮的水平位置 */
  padding: 10px 20px;
  background-color: white; /* 改变按钮颜色 */
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  z-index: 10; /* 确保按钮位于其他元素之上 */
}
.node{
    font-size: 25px; 
    text-align: center;
    height: 50px;
    background-color:#b3e19d;
}
.inputvalue{
  display: flex;
  margin-top: 10px;
  margin-left: 7px;
  font-size: 20px;
  font-weight: 500;

}
</style>