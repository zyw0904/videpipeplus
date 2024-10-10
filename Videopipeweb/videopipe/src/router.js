// src/router.js
import {createRouter, createWebHashHistory} from 'vue-router';
//import App from './App.vue';
import InferDeivces from './components/InferDeivces.vue';
import VideoInfer from './components/VideoInfer.vue';

const routes = [
    {path: '/', name: 'home', component: InferDeivces},
    {path: '/another', name:'another', component: VideoInfer},
]

const router = createRouter({
    history: createWebHashHistory(),
    routes
})

export default router