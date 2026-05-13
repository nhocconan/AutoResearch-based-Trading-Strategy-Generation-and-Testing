#!/usr/bin/env python3
# 4h_ThreeLineBreak_Trend_Filter
# Hypothesis: Three Line Break (TLB) detects sustained momentum without whipsaw.
# Long when TLB shows bullish reversal (green brick after red) + price above 4h EMA50.
# Short when TLB shows bearish reversal (red brick after green) + price below 4h EMA50.
# EMA50 filter ensures alignment with medium-term trend, reducing counter-trend trades.
# Works in bull markets (captures uptrend continuation) and bear markets (captures downtrend continuation).
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "4h_ThreeLineBreak_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # Calculate Three Line Break
    tlb = np.zeros(n, dtype=int)  # 1 for bullish, -1 for bearish
    line_heights = []  # track closing prices of each line
    current_line = 1  # start with bullish
    line_start_idx = 0

    for i in range(n):
        if i == 0:
            line_heights.append(close[i])
            tlb[i] = current_line
            continue

        if current_line == 1:  # bullish line
            if close[i] > line_heights[-1]:
                line_heights.append(close[i])
            elif close[i] < line_heights[-3] if len(line_heights) >= 3 else False:
                # reverse to bearish
                current_line = -1
                line_heights = [close[i]]
                line_start_idx = i
            tlb[i] = current_line
        else:  # bearish line
            if close[i] < line_heights[-1]:
                line_heights.append(close[i])
            elif close[i] > line_heights[-3] if len(line_heights) >= 3 else False:
                # reverse to bullish
                current_line = 1
                line_heights = [close[i]]
                line_start_idx = i
            tlb[i] = current_line

    # EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # start after EMA50 warmup
        if position == 0:
            # LONG: TLB bullish reversal (current bullish, previous bearish) + above EMA50
            if tlb[i] == 1 and tlb[i-1] == -1 and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TLB bearish reversal (current bearish, previous bullish) + below EMA50
            elif tlb[i] == -1 and tlb[i-1] == 1 and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TLB bearish reversal or price below EMA50
            if tlb[i] == -1 and tlb[i-1] == 1 or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TLB bullish reversal or price above EMA50
            if tlb[i] == 1 and tlb[i-1] == -1 or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals