"""CNN 学習 (M4, 任意).

速度勝負（高速クリッカー）の見せ場が必要な場合のみ用意する軽量バックエンド。
~2s/回の Nemotron では「人類1分チャレンジ」に間に合わないため、サブミリ秒で
判定する CNN を別に持ち、Nemotron は解説役に回す。速度勝負が不要なら省略可。

MobileNetV3-small の分類ヘッドを 2 クラス (A/B) に付け替えて学習する。
data/train で学習し data/val で評価、ベスト重みを保存する（重みはコミットしない）。
学習ログ・チャートは results/ に出力する。

TODO(M4): torchvision MobileNetV3-small + ImageFolder で実装。
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("M4: CNN training not implemented yet")


if __name__ == "__main__":
    main()
