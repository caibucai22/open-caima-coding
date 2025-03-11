昨天有群友，在交流群【群号：392784757】里提到了这个需求，进行实现一下
![image.png](https://i-blog.csdnimg.cn/blog_migrate/a2f7827f5be656390326377398cb33f2.png)

V10官方代码结构相较于V8 稍微复杂一些
![image.png](https://i-blog.csdnimg.cn/blog_migrate/7ff619083642d7b4afeb124fa3ad583d.png)
yolov10 是基于 v8的代码完成开发，yolov10进行了继承来简化代码开发
因此V10的代码修改 基本和V8 这篇一致
[https://blog.csdn.net/csy1021/article/details/134406419](https://blog.csdn.net/csy1021/article/details/134406419)

但存在一些不同，会在下面提到

# 版本环境

YOLOv10 2024.07.01 版本

# 修改

## trainer.py

### 1 添加save_metrics_per_class()

在save_metrics函数后面，添加下面的 save_metrics_per_class 函数

```python
def save_metrics_per_class(self, box):

    """Saves training metrics per class to a CSV file."""

    # ap ap50 p r 提示作用
    keys = ['ap', 'ap50', 'p', 'r']
    n = 4 + 1  # number of cols

    for i in box.ap_class_index:
        cur_class = self.model.names[box.ap_class_index[i]]
        save_path = self.save_dir.joinpath("result_" + cur_class + ".csv")
        vals = [box.ap[i], box.ap50[i], box.p[i], box.r[i]]
        s = '' if save_path.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header

        with open(save_path, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')
```

### 2 validate() 修改

```python
def validate(self):
    """
    Runs validation on test set using self.validator.

    The returned dict is expected to contain "fitness" key.
    """
    # metrics = self.validator(self)
    metrics,box = self.validator(self)
    fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
    if not self.best_fitness or self.best_fitness < fitness:
        self.best_fitness = fitness
    # return metrics, fitness
    return metrics, fitness,box
```

找到【这里比v8的判断要多】

```python
if (self.args.val and (((epoch+1) % self.args.val_period == 0) or (self.epochs - epoch) <= 10)) \
                    or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
```

修改为

```python
if (self.args.val and (((epoch+1) % self.args.val_period == 0) or (self.epochs - epoch) <= 10)) \
                    or final_epoch or self.stopper.possible_stop or self.stop:
                    # self.metrics, self.fitness = self.validate()
                    self.metrics, self.fitness,box = self.validate()
```

### 3 找到  self.save_metrics

在
 self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr}) 
 后面添加调用
self.save_metrics_per_class(box)

## validator.py

找到 stats = self.get_stats() 
改为 stats,box = self.get_stats()

找到 return {k: round(float(v), 5) for k, v in results.items()}
改为 return {k: round(float(v), 5) for k, v in results.items()}, box

## val.py

### get_stats() 【注意与v8不同】

```python
def get_stats(self):
    """Returns metrics statistics and results dictionary."""
    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
    # if len(stats) and stats["tp"].any():
    # if len(stats) and stats[0].any():
    if len(stats) :
        self.metrics.process(**stats)
    self.nt_per_class = np.bincount(
        stats["target_cls"].astype(int), minlength=self.nc
    )  # number of targets per class
    # return self.metrics.results_dict
    return self.metrics.results_dict,self.metrics.box
```

# save_metrics_per_class() 函数 【注意与v8不同】

<img src="https://i-blog.csdnimg.cn/blog_migrate/bbce94337d898284d2d3768c0d5d9ac2.png" alt="image.png" style="zoom:80%;" />
可以看到支持的指标有 all_ap （可用来计算其他ap指标），map，map50，f1，p ap，r  mr ...
我在函数中使用的是 ap，ap50，p，r，需要其他的可以再添加
==注意：添加指标，使用的是 . 而不是 ["xxxx"] 如 box.ap[i] 而不是 box['ap'][i]==

```python
def save_metrics_per_class(self, box):

    """Saves training metrics per class to a CSV file."""

    # ap ap50 p r 提示作用
    keys = ['ap', 'ap50', 'p', 'r']
    n = 4 + 1  # number of cols

    for i in box.ap_class_index:
        cur_class = self.model.names[box.ap_class_index[i]]
        save_path = self.save_dir.joinpath("result_" + cur_class + ".csv")
        vals = [box.ap[i], box.ap50[i], box.p[i], box.r[i]]
        s = '' if save_path.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header

        with open(save_path, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')
```

# 注意！不同点

```python
def get_stats(self):
    """Returns metrics statistics and results dictionary."""
    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
    # if len(stats) and stats["tp"].any(): # v10
    # if len(stats) and stats[0].any(): # v8
    if len(stats) : # 修改后
        self.metrics.process(**stats)
    self.nt_per_class = np.bincount(
        stats["target_cls"].astype(int), minlength=self.nc
    )  # number of targets per class
    # return self.metrics.results_dict
    return self.metrics.results_dict,self.metrics.box
```

v10
![image.png](https://i-blog.csdnimg.cn/blog_migrate/3fc5644710595c2dc6bbdd1d67340029.png)
v8
![image.png](https://i-blog.csdnimg.cn/blog_migrate/fdd4c3fc5448937f876a90b6b0ccf10d.png)

![image.png](https://i-blog.csdnimg.cn/blog_migrate/077a3402bf579e2fa79d018e953aaedf.png)

如果不修改 这个判断条件

```python
if len(stats) and stats["tp"].any(): # v10
# if len(stats) and stats[0].any(): # v8 仅作对比
if len(stats) : # 修改后
```

可能会出现 前几次 epoch 数据不记录的问题 【这里也可能是和我的数据集有关，我测试了几次，增加batch-size 发现仍然 stats["tp"] 仍然全为 false 过不了，后面epoch会正常 】这里大家可以自行测试后决定，如果正常，就不需要改

# 其他
增加训练过程各类指标打印（可选，默认开启是有条件的）
val.py  
找到 print_results() 函数 在
LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())) 后面
添加

```python
for i, c in enumerate(self.metrics.ap_class_index):
    LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))
```

有问题，欢迎留言、进群讨论或私聊：【群号：392784757】

