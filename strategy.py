#!/usr/bin/env python3
# 6h_Aroon_Signal_WeeklyTrend_Confirm
# Hypothesis: Aroon oscillator (25-period) identifies strong trends, with weekly trend filter (price vs 50-week SMA) and volume confirmation (1.5x 20-period average). Aroon > 50 indicates uptrend strength, < -50 indicates downtrend. Works in bull/bear by following higher timeframe trend. Target: 15-30 trades/year.

name = "6h_Aroon_Signal_WeeklyTrend_Confirm"
timeframe = "6h"
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
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values

    # Calculate 50-week SMA for trend filter
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)

    # Aroon oscillator (25-period)
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period, n):
        # Periods since highest high
        highest_high_idx = i - np.argmax(high[i-period:i+1])
        # Periods since lowest low
        lowest_low_idx = i - np.argmin(low[i-period:i+1])
        aroon_up[i] = ((period - highest_high_idx) / period) * 100
        aroon_down[i] = ((period - lowest_low_idx) / period) * 100
    
    aroon_osc = aroon_up - aroon_down  # Range: -100 to +100
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after weekly SMA warmup
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(aroon_osc[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Aroon > 50 (uptrend) + price > 50-week SMA + volume confirmation
            if (aroon_osc[i] > 50 and 
                close[i] > sma_50_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Aroon < -50 (downtrend) + price < 50-week SMA + volume confirmation
            elif (aroon_osc[i] < -50 and 
                  close[i] < sma_50_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Aroon <= 0 (trend weakening) or price < 50-week SMA
            if (aroon_osc[i] <= 0 or 
                close[i] < sma_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Aroon >= 0 (trend weakening) or price > 50-week SMA
            if (aroon_osc[i] >= 0 or 
                close[i] > sma_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals