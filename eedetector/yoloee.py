'''=================================================
@Project -> File   ：webserver -> yoloee
@IDE    ：PyCharm
@Author ：lechatelia
@Date   ：2019/12/14 11:46
@Desc   ：
=================================================='''
import argparse
import time
import matplotlib.pyplot as plt
import cv2
from  PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='used for detect foreign objects')
    parser.add_argument('--checkpoint', default="/data/codes/yolov3ee/save/model_yoloee.pt", type=str, help='pretrained model name')
    parser.add_argument('--cfg', default="/data/codes/yolov3/cfg/yolov3ee.vfg", type=str, help='cfg file path')
    parser.add_argument('--save_images', default=True, type=bool, help='whether to save predicted image')
    parser.add_argument('--conf-threshold', default=0.5, type=float, help='confidence threshod of predicted value')
    parser.add_argument('--nms-threshold', default=0.5, type=float, help='  iou threshod of nms ')
    parser.add_argument('--gpu_number', default=2, type=int, help='which one gpu to use')
    parser.add_argument('--output_dir', default='./output', type=str, help='where to save predictions')
    args = parser.parse_args()
    return args

class YoloEE():
    '''yolo detector for detect foreign objects to inspect transmission lines'''
    def __init__(self,gpu_number=None,img_size=608):
        self.args = parse_args()
        self.img_size = img_size
        if gpu_number is not None:
            self.args.gpu_number = gpu_number
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        self.device = torch.device("cuda:{}".format(self.args.gpu_number) if torch.cuda.is_available() else "cpu")
        print(self.device)
        checkpoints = torch.load(self.args.checkpoint)
        self.model = checkpoints['model']
        self.model = self.model.to(self.device)
        self.model.eval()
        self.cfg = checkpoints['cfg']
        self.classlist = checkpoints['classes']
        self.colors = [[random.randint(0,255) for _ in range(3)]for _ in range(len(self.classlist))]

    def detect(self,image):
        if not image.mode=='RGB':
            image.convert('RGB')
        image = np.asarray(image)
        img, _, _, _ = letterbox(image,height=self.img_size)
        img = np.ascontiguousarray(img.transpose(2,0,1), dtype=np.float32)
        img /= 255.0
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        pred, _ = self.model(img)
        detections = non_max_suppression(pred,self.args.conf_threshold, self.args.nms_threshold)[0]
        results = []
        if detections is not None and len(detections)>0:
            scale_coords(self.img_size, detections[:4], image.shape).round()
            # Draw bounding boxes and labels of detections

            for *xyxy, conf, cls_conf, cls in detections:
                # Add bbox to the image
                label = '%s %.2f' % (self.classlist[int(cls)], conf)
                plot_one_box(xyxy, image, label=label, color=self.colors[int(cls)])
                results.append({"xyxy":[x.item() for x in xyxy], "confidence": "%0.2f"%(100*conf.item()), "classname": self.classlist[int(cls)] })

        return image, results


    def test(self, path):
        image = Image.open(path)
        img, results = self.detect(image)
        img = Image.fromarray(img)
        plt.imshow(img)
        plt.show()
        return results

if __name__ == "__main__":
    detector = YoloEE()
    results = detector.test('/data/codes/yolov3ee/samples/DiaoChe00003.jpg')
    print(results)


