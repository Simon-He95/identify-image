<script setup lang="ts">
import identifyImage from '../../dist'
onMounted(async () => {
  await identifyImage('../public/favicon.svg')
})
</script>

<template>
  <div />
</template>

<style scoped>
</style>
