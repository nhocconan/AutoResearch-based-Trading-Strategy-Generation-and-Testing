#!/usr/bin/env python3
# 4h_1d_keltner_breakout_volume_trend_v1
# Hypothesis: 4-hour strategy using 1-day EMA100 for trend direction and 1-day Keltner breakout for entries, with volume confirmation.
# Works in bull/bear by requiring alignment with the 1d trend and confirming with volume to avoid false breakouts.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "4h_1d_keltner_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and Keltner
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA100 for trend direction
    ema100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Previous 1d bar's data for Keltner
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # ATR(20) for Keltner channels
    tr1 = np.abs(prev_high_1d - prev_low_1d)
    tr2 = np.abs(prev_high_1d - prev_close_1d)
    tr3 = np.abs(prev_low_1d - prev_close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner channels (2x ATR)
    upper_keltner = prev_close_1d + 2 * atr20
    lower_keltner = prev_close_1d - 2 * atr20
    
    # Align to 4h timeframe
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema100_1d_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price > EMA100 (uptrend) AND close breaks above upper Keltner with volume
        if (close[i] > ema100_1d_aligned[i] and close[i] > upper_keltner_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price < EMA100 (downtrend) AND close breaks below lower Keltner with volume
        elif (close[i] < ema100_1d_aligned[i] and close[i] < lower_keltner_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite Keltner band
        elif position == 1 and close[i] < lower_keltner_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > upper_keltner_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals