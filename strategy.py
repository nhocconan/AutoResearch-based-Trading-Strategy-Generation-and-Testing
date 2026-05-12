#!/usr/bin/env python3
"""
4h_TRIX_Momentum_Volume_Spike
Hypothesis: TRIX (1-period ROC of EMA) captures momentum shifts. Combined with volume spike (>2x 20-period average) and price above/below EMA50 for trend filter. Designed for 4h timeframe to capture medium-term momentum moves in both bull and bear markets. Low trade frequency expected due to strict volume and momentum conditions.
"""

name = "4h_TRIX_Momentum_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate TRIX: 1-period ROC of triple EMA (15-period EMA applied 3 times)
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    # TRIX = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix_raw = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix_raw.fillna(0).values  # Replace NaN with 0 for stability

    # EMA50 for trend filter
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start from 20 to have enough data for indicators
        # Skip if any required data is invalid
        if np.isnan(trix[i]) or np.isnan(ema50[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Positive TRIX momentum + price above EMA50 + volume spike
            if (trix[i] > 0 and 
                close[i] > ema50[i] and 
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Negative TRIX momentum + price below EMA50 + volume spike
            elif (trix[i] < 0 and 
                  close[i] < ema50[i] and 
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Negative TRIX momentum or price below EMA50
            if (trix[i] < 0 or close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Positive TRIX momentum or price above EMA50
            if (trix[i] > 0 or close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals