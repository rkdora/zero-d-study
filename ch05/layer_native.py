# 乗算レイヤ
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 順伝播
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    # 逆伝播
    def backward(self, dout):
        # xとyをひっくり返す
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy
