import * as mobilenet from '@tensorflow-models/mobilenet'
import '@tensorflow/tfjs-backend-webgl'
import type { Prediction } from './types'

export function identifyImage(src: string): Promise<Prediction[]> {
  // eslint-disable-next-line no-async-promise-executor
  return new Promise(async (resolve) => {
    const img = new Image()
    img.src = src
    const model = await mobilenet.load()
    const predictions = await model.classify(img)
    resolve(predictions)
  })
}
