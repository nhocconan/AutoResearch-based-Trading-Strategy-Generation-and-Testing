#!/usr/bin/env python3
name = "6h_WeeklyPivot_R3S3_Breakout_TrendVolume"
timeframe = "6h"
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
    
    # Get weekly data for pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Weekly OHLC
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Previous week's values (avoid look-ahead)
    prev_high_w = np.roll(high_w, 1)
    prev_low_w = np.roll(low_w, 1)
    prev_close_w = np.roll(close_w, 1)
    prev_high_w[0] = high_w[0]
    prev_low_w[0] = low_w[0]
    prev_close_w[0] = close_w[0]
    
    # Weekly pivot and levels (based on previous week)
    pp_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    r1_w = 2 * pp_w - prev_low_w
    s1_w = 2 * pp_w - prev_high_w
    r2_w = pp_w + (prev_high_w - prev_low_w)
    s2_w = pp_w - (prev_high_w - prev_low_w)
    r3_w = prev_high_w + 2 * (pp_w - prev_low_w)
    s3_w = prev_low_w - 2 * (prev_high_w - pp_w)
    r4_w = prev_high_w + 3 * (pp_w - prev_low_w)
    s4_w = prev_low_w - 3 * (prev_high_w - pp_w)
    
    # Daily EMA34 for trend filter
    close_d = df_d['close'].values
    ema_34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly pivot levels to 6h timeframe
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    ema_34_aligned = align_htf_to_ltf(prices, df_d, ema_34_d)
    
    # Volume filter: 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_w_aligned[i]) or np.isnan(s3_w_aligned[i]) or 
            np.isnan(r4_w_aligned[i]) or np.isnan(s4_w_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: Price breaks above weekly R3 with volume, AND above daily EMA34 (bullish bias)
            if (close[i] > r3_w_aligned[i] and 
                volume_surge and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S3 with volume, AND below daily EMA34 (bearish bias)
            elif (close[i] < s3_w_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: 
            # For long: price returns below weekly S3 or trend turns bearish
            # For short: price returns above weekly R3 or trend turns bullish
            if position == 1:
                if (close[i] < s3_w_aligned[i]) or (close[i] < ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (close[i] > r3_w_aligned[i]) or (close[i] > ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals