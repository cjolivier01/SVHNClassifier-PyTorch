import argparse

import torch
from model import Model
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint, e.g. ./logs/model-100.pth')
# parser.add_argument('input', type=str, help='path to input image')


def _infer(path_to_checkpoint_file, path_to_input_image):
    model = Model()
    model.restore(path_to_checkpoint_file)
    model.cuda()
    model.eval()

    with torch.no_grad():
        transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image = Image.open(path_to_input_image)
        image = image.convert('RGB')
        image = transform(image)
        images = image.unsqueeze(dim=0).cuda()

        (
            length_logits,
            digit1_logits,
            digit2_logits,
            digit3_logits,
            digit4_logits,
            digit5_logits,
        ) = model(images)

        length_value, length_prediction = length_logits.max(1)
        digit1_value, digit1_prediction = digit1_logits.max(1)
        digit2_value, digit2_prediction = digit2_logits.max(1)
        digit3_value, digit3_prediction = digit3_logits.max(1)
        digit4_value, digit4_prediction = digit4_logits.max(1)
        digit5_value, digit5_prediction = digit5_logits.max(1)

        print("length:", length_prediction.item(), "value:", length_value.item())
        print('digits:', digit1_prediction.item(), digit2_prediction.item(), digit3_prediction.item(), digit4_prediction.item(), digit5_prediction.item())
        print(
            "values:",
            digit1_value.item(),
            digit2_value.item(),
            digit3_value.item(),
            digit4_value.item(),
            digit5_value.item(),
        )
        all_digits = [
            digit1_prediction.item(),
            digit2_prediction.item(),
            digit3_prediction.item(),
            digit4_prediction.item(),
            digit5_prediction.item(),
        ]
        running = 0
        for i in range(length_prediction.item()):
            running *= 10
            running += all_digits[i]
        print(f"Final prediction: {running}")


def main(args):
    # path_to_checkpoint_file = args.checkpoint
    # path_to_input_image = args.input

    path_to_checkpoint_file = "pretrained/svhnc/model-65000.pth"
    path_to_input_image = "/home/colivier/src/hm/test-19.png"

    _infer(path_to_checkpoint_file, path_to_input_image)


if __name__ == '__main__':
    main(parser.parse_args())
