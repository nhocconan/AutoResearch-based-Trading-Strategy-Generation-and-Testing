#!/usr/bin/env python3
name = "12h_TriangleBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume filter: volume > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 12h triangle: 20-period high/low (Donchian)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if 1d trend or volume data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Triangle breakout: break above 20-period high or below 20-period low
        if position == 0:
            # Long: break above 20-period high with 1d uptrend and volume
            if (high[i] > high_20[i] and 
                close[i] > high_20[i] and
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_ma_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below 20-period low with 1d downtrend and volume
            elif (low[i] < low_20[i] and 
                  close[i] < low_20[i] and
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_ma_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below 20-period low or trend reversal
            if (low[i] < low_20[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above 20-period high or trend reversal
            if (high[i] > high_20[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals