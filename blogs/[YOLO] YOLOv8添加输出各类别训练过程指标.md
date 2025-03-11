
场景：当前YOLOv8训练过程中输出的是针对所有类别的各项指标，具体数据保存在 results.csv 这个文件中。有时候，我们想要得到具体类别训练过程中的指标情况，此时就没有办法了，使用val.py 也只是输出最终的一个各类指标，无法得到训练过程的指标数据，针对这一需求，进行修改，最终可以得到每一类各项指标的一个csv文件，方便后续作图和实验对比

具体修改后效果如下：

<div align=center><img src="https://i-blog.csdnimg.cn/blog_migrate/618230e2305e07f6c07e4cdd93ee88d7.png"></div>


各个类别训练过程的指标被保存到相应的result_xxx.csv 文件中



# 版本环境

YOLOv8 8月份版本，最新版本在测

# 修改
## trainer.py 

1 添加一个 save_metrics_per_class 函数，放到save_metrics函数后面

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

2 validate函数修改

```python
def validate(self):
    """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
    # metrics = self.validator(self) 原始版本
    metrics, box = self.validator(self)
    fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
    if not self.best_fitness or self.best_fitness < fitness:
        self.best_fitness = fitness
    # return metrics, fitness 原始版本
    return metrics, fitness, box
```

找到

```python
if self.args.val or final_epoch:
    self.metrics, self.fitness = self.validate()
```

修改为

```python
if self.args.val or final_epoch:
    self.metrics, self.fitness, box = self.validate()
```

## validator.py

找到 stats = self.get_stats() 

改为 stats,box = self.get_stats()



找到 return {k: round(float(v), 5) for k, v in results.items()}

改为 return {k: round(float(v), 5) for k, v in results.items()}, box



## val.py 

get_stats()

找到 return self.metrics.results_dict

改为 return self.metrics.results_dict,self.metrics.box



save_metrics_per_class 函数

![image-20231114191804472](https://i-blog.csdnimg.cn/blog_migrate/4e4d7e4bc9de6566cd4d78e5fbadb90b.png)

可以看到支持的指标有 all_ap （可用来计算其他ap指标），ap，ap50，f1，p，r

我在函数中使用的是 ap，ap50，p，r

==注意：添加指标，使用的是 . 而不是 ["xxxx"] 如 box.ap[i] 而不是 box['ap']\[i]==

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

## 最后
 在 trainer.py 找到
 self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr}) 
 在后面添加调用

self.save_metrics_per_class(box)

即可




# 其他 

增加训练过程各类指标打印（可选，默认开启是有条件的）

val.py  找到 print_results() 函数 在

LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())) 后面

添加

```python
for i, c in enumerate(self.metrics.ap_class_index):
    LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))
```


有问题，欢迎留言、进群讨论或私聊：【群号：392784757】
