#!/usr/bin/env python3
# 6h_LongOnly_Pullback_To_EMA21_With_Volume_Confirmation
# Hypothesis: In trending markets, price pulls back to the 21-period EMA on 6h charts.
# Long entries occur when price touches EMA21 with bullish momentum (close > open) and volume > 1.5x 20-period average.
# Exit on close below EMA21 or bearish engulfing candle. Works in bull trends (buying dips) and avoids shorts in bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_LongOnly_Pullback_To_EMA21_With_Volume_Confirmation"
timeframe = "6h"
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
    open_ = prices['open'].values
    volume = prices['volume'].values

    # Calculate EMA21 on close
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long

    for i in range(21, n):
        # Skip if any required value is NaN
        if np.isnan(ema_21[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG ENTRY: price touches EMA21 (within 0.5%), bullish candle, volume spike
            ema_touch = low[i] <= ema_21[i] * 1.005 and high[i] >= ema_21[i] * 0.995
            bullish_candle = close[i] > open_[i]
            volume_spike = volume[i] > vol_avg_20[i] * 1.5

            if ema_touch and bullish_candle and volume_spike:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close below EMA21 or bearish engulfing candle
            bearish_engulfing = close[i] < open_[i] and open_[i] < close[i-1] and close[i] < open_[i-1]
            if close[i] < ema_21[i] or bearish_engulfing:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25

    return signals