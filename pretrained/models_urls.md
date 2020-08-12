## Pretrained Models

Here, we provide the pretrained models of PAN models on Something-Something-V1 & V2 datasets. Recognizing actions in these datasets requires strong temporal modeling ability, as many action classes are symmetrical. PAN achieves state-of-the-art performance on these datasets. Notably, our method even surpasses optical flow based methods while with only RGB frames as input.

### Something-Something-V1

<div align="center">
<table>
<thead>
<tr>
<th align="center">Model</th>
<th align="center">Backbone</th>
<th align="center">FLOPs * views</th>
<th align="center">Val Top1</th>
<th align="center">Val Top5</th>
<th align="center">Checkpoints</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">PAN<sub>Lite</sub></td>
<td align="center" rowspan="3">ResNet-50</td>
<td align="center">35.7G * 1</td>
<td align="center">48.0</td>
<td align="center">76.1</td>
<td align="center" rowspan="3">[<a href="https://drive.google.com/drive/folders/1Be3o8pesPrk8uEoBNd4OW6qtKkeXoVBU?usp=sharing">Google Drive</a>] or [<a href="https://share.weiyun.com/F2PJnUiE" rel="nofollow">Weiyun</a>]</td>
</tr>
<tr>
<td align="center">PAN<sub>Full</sub></td>
<td align="center">67.7G * 1</td>
<td align="center">50.5</td>
<td align="center">79.2</td>
</tr>
<tr>
<td align="center">PAN<sub>En</sub></td>
<td align="center">(46.6G+88.4G) * 2</td>
<td align="center">53.4</td>
<td align="center">81.1</td>
</tr>
<tr>
<td align="center">PAN<sub>En</sub></td>
<td align="center">ResNet-101</td>
<td align="center">(85.6G+166.1G) * 2</td>
<td align="center">55.3</td>
<td align="center">82.8</td>
<td align="center">[<a href="https://drive.google.com/drive/folders/1sD242c3rkhIjTPxJiyPnXRx7zKLnA12b?usp=sharing">Google Drive</a>] or [<a href="https://share.weiyun.com/bqouigPA" rel="nofollow">Weiyun</a>]</td>
</tr>
</tbody>
</table>
</div>

### Something-Something-V2

<div align="center">
<table>
<thead>
<tr>
<th align="center">Model</th>
<th align="center">Backbone</th>
<th align="center">FLOPs * views</th>
<th align="center">Val Top1</th>
<th align="center">Val Top5</th>
<th align="center">Checkpoints</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">PAN<sub>Lite</sub></td>
<td align="center" rowspan="3">ResNet-50</td>
<td align="center">35.7G * 1</td>
<td align="center">60.8</td>
<td align="center">86.7</td>
<td align="center" rowspan="3">[<a href="https://drive.google.com/drive/folders/1dqMphONnGLBxhArMazr281Ix4qOX58MI?usp=sharing">Google Drive</a>] or [<a href="https://share.weiyun.com/oXQY3rx7" rel="nofollow">Weiyun</a>]</td>
</tr>
<tr>
<td align="center">PAN<sub>Full</sub></td>
<td align="center">67.7G * 1</td>
<td align="center">63.8</td>
<td align="center">88.6</td>
</tr>
<tr>
<td align="center">PAN<sub>En</sub></td>
<td align="center">(46.6G+88.4G) * 2</td>
<td align="center">66.2</td>
<td align="center">90.1</td>
</tr>
<tr>
<td align="center">PAN<sub>En</sub></td>
<td align="center">ResNet-101</td>
<td align="center">(85.6G+166.1G) * 2</td>
<td align="center">66.5</td>
<td align="center">90.6</td>
<td align="center">[<a href="https://drive.google.com/drive/folders/1NL-CNQPLfV3abnJDMcz3AF4WFsH7LDhN?usp=sharing">Google Drive</a>] or [<a href="https://share.weiyun.com/oSOgog0J" rel="nofollow">Weiyun</a>]</td>
</tr>
</tbody>
</table>
</div>

