#!/usr/bin/env python3
name = "4h_Pivot_Breakout_Supertrend_1d"
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
    
    # Get 1D data for daily pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Pivot support/resistance levels
    R1 = pivot + (range_val * 1.0 / 3.0)
    S1 = pivot - (range_val * 1.0 / 3.0)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1D EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Supertrend (ATR=10, multiplier=3.0)
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    supertrend = np.zeros(n)
    dir = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    dir[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            dir[i] = 1
        else:
            dir[i] = -1
        
        if dir[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
    
    # Volume ratio (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(supertrend[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above R1 with volume surge and above Supertrend/uptrend
            if (close[i] > R1_aligned[i] and 
                volume_surge and 
                dir[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume surge and below Supertrend/downtrend
            elif (close[i] < S1_aligned[i] and 
                  volume_surge and 
                  dir[i] == -1):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Price returns below pivot or trend turns bearish
                if (close[i] < pivot_aligned[i]) or (dir[i] == -1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Price returns above pivot or trend turns bullish
                if (close[i] > pivot_aligned[i]) or (dir[i] == 1):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals