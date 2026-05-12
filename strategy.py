#!/usr/bin/env python3
# 160101: 4h_Price_Action_Pullback_1dTrend_Volume
# Hypothesis: Buy pullbacks to value area in trending markets using 1d EMA50 trend filter, 4h support/resistance from prior day's range, and volume confirmation. Works in bull/bear by following higher timeframe trend. Designed for low trade frequency (target: 20-40/year) to minimize fee drag.

name = "4h_Price_Action_Pullback_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Prior day's high/low for value area (support/resistance)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Pullback to support in uptrend with volume
            if (close[i] <= low_1d_aligned[i] * 1.005 and  # Near prior day low
                close[i] > ema_50_1d_aligned[i] and       # Above 1d EMA50 (uptrend)
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback to resistance in downtrend with volume
            elif (close[i] >= high_1d_aligned[i] * 0.995 and  # Near prior day high
                  close[i] < ema_50_1d_aligned[i] and        # Below 1d EMA50 (downtrend)
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below 1d EMA50 (trend change) or strong rejection
            if close[i] < ema_50_1d_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above 1d EMA50 (trend change) or strong rejection
            if close[i] > ema_50_1d_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals