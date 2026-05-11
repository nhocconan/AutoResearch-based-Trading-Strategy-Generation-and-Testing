#!/usr/bin/env python3
name = "12h_1w_1d_Camarilla_R3S3_Breakout_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1W and 1D data (HTF)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1W Trend Filter ===
    # EMA50 on weekly close for long-term trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1D Camarilla Pivot Levels ===
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
    
    # Pivot and levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    R3 = pivot + (range_val * 1.1 / 2.0)
    R1 = pivot + (range_val * 1.1 / 6.0)
    S1 = pivot - (range_val * 1.1 / 6.0)
    S3 = pivot - (range_val * 1.1 / 2.0)
    
    # Align to 12h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # === 1D EMA34 for medium-term trend ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_surge = vol_ratio[i] > 1.5
        
        # Long-term trend from weekly
        long_term_up = close[i] > ema_50_1w_aligned[i]
        long_term_down = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume surge and above weekly EMA50 (bullish long-term trend)
            if (close[i] > R3_aligned[i] and 
                volume_surge and 
                long_term_up):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume surge and below weekly EMA50 (bearish long-term trend)
            elif (close[i] < S3_aligned[i] and 
                  volume_surge and 
                  long_term_down):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: Price returns below R1 or breaks below weekly EMA50 (trend change)
                if (close[i] < R1_aligned[i]) or (not long_term_up):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Price returns above S1 or breaks above weekly EMA50 (trend change)
                if (close[i] > S1_aligned[i]) or (not long_term_down):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals