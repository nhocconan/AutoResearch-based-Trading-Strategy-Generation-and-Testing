#!/usr/bin/env python3
# 1h_pullback_volume_sr_4h1d_v1
# Hypothesis: Buy pullbacks in strong uptrends on 4h with volume confirmation on 1h; sell rallies in strong downtrends on 4h with volume confirmation.
# Uses 4h EMA(50) for trend direction, 1h EMA(20) for pullback/retrace entries, and volume > 1.5x average for confirmation.
# Designed to work in both bull and bear markets by trading with the 4h trend and using volume to confirm momentum.
# Target: 15-37 trades/year (60-150 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_pullback_volume_sr_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend: EMA(50) on 4h close
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = np.zeros(len(close_4h))
    ema_4h[0] = close_4h[0]
    alpha_4h = 2.0 / (50 + 1)
    for i in range(1, len(close_4h)):
        ema_4h[i] = alpha_4h * close_4h[i] + (1 - alpha_4h) * ema_4h[i-1]
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h EMA(20) for pullback entries
    ema_20 = np.zeros(n)
    ema_20[0] = close[0]
    alpha_20 = 2.0 / (20 + 1)
    for i in range(1, n):
        ema_20[i] = alpha_20 * close[i] + (1 - alpha_20) * ema_20[i-1]
    
    # Volume confirmation: 1h volume > 1.5x 20-period average
    vol_ma_20 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(ema_4h_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price crosses below 1h EMA(20) or trend turns bearish
            if close[i] < ema_20[i] or close[i] < ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1h EMA(20) or trend turns bullish
            if close[i] > ema_20[i] or close[i] > ema_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price > 4h EMA (uptrend) and pullback to 1h EMA(20) with volume
            if close[i] > ema_4h_aligned[i] and close[i] <= ema_20[i] and close[i-1] > ema_20[i-1] and vol_ok:
                position = 1
                signals[i] = 0.20
            # Enter short: price < 4h EMA (downtrend) and retrace to 1h EMA(20) with volume
            elif close[i] < ema_4h_aligned[i] and close[i] >= ema_20[i] and close[i-1] < ema_20[i-1] and vol_ok:
                position = -1
                signals[i] = -0.20
    
    return signals