# identify-image
根据图片识别最大可能的东西

# Install
```
npm i @simon_he/identify-image
```

# Usage
```
import identifyImage from '@simon_he/identify-image'
const info = await identifyImage('./person.jpg')
```

# Result
```js
[
  { className: 'analog clock', probability: 0.13620825111865997 },
  { className: 'magnetic compass', probability: 0.09165980666875839 },
  { className: 'guillotine', probability: 0.04591737687587738 }
]
```

# License
MIT

# Powered by
- [tensorflow.js](https://github.com/tensorflow/tfjs-models)

## :coffee: 
<a href="https://github.com/Simon-He95/sponsor" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" style="height: 51px !important;width: 217px !important;" ></a>

<span><div align="center">![sponsors](https://www.hejian.club/images/sponsors.jpg)</div></span>
