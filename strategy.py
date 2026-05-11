#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_v1
Hypothesis: Combines 4h Donchian breakouts with volume confirmation and 1d trend filter.
Breakouts above 20-period high with volume > 1.5x average and bullish 1d trend = long.
Breakouts below 20-period low with volume > 1.5x average and bearish 1d trend = short.
Designed to capture trending moves while avoiding false breakouts in ranging markets.
Targets 30-60 trades per year (120-240 over 4 years) on 4h timeframe.
"""

name = "4h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Donchian Channel (20-period) ===
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # === Volume Confirmation (1.5x average volume) ===
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    volume_ratio = np.divide(volume, vol_ma, out=np.full(n, np.nan), where=vol_ma!=0)
    
    # === 1d Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(lookback, vol_period, 34)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(volume_ratio[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + volume confirmation + bullish 1d trend
            if (close[i] > highest_high[i] and 
                volume_ratio[i] > 1.5 and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + volume confirmation + bearish 1d trend
            elif (close[i] < lowest_low[i] and 
                  volume_ratio[i] > 1.5 and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low or loss of 1d trend
            if close[i] < lowest_low[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: close above Donchian high or loss of 1d trend
            if close[i] > highest_high[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals