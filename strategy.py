#!/usr/bin/env python3
# 12h_1d_1w_Camarilla_R3_S3_Breakout_Trend_Filter_v1
# Hypothesis: Combines Camarilla pivot levels from daily timeframe with weekly trend filter
# and volume confirmation on 12h timeframe. Goes long when price breaks above R3 level with
# volume confirmation and weekly trend is up; goes short when price breaks below S3 level with
# volume confirmation and weekly trend is down. Uses R3/S3 levels for stronger breakout signals
# to reduce trade frequency. Weekly trend filter avoids counter-trend trades in choppy markets.
# Designed for low trade frequency (target: 50-150 total over 4 years) by requiring multiple
# confluence factors: Camarilla breakout, volume spike, and weekly trend alignment.

name = "12h_1d_1w_Camarilla_R3_S3_Breakout_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Camarilla Pivot Levels from 1d data ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3/S3 are the strongest breakout levels
    r3 = pivot + (range_1d * 1.1)
    s3 = pivot - (range_1d * 1.1)
    
    # Align Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- Weekly Trend Filter (EMA34 on 1w close) ---
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_1w_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price breaks above R3 with volume, weekly trend up
            if (close[i] > r3_aligned[i] and 
                volume_spike and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume, weekly trend down
            elif (close[i] < s3_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla level break or loss of weekly trend
            if position == 1:
                # Exit long: price breaks below S3 or weekly trend turns down
                if (close[i] < s3_aligned[i] or 
                    close[i] < ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R3 or weekly trend turns up
                if (close[i] > r3_aligned[i] or 
                    close[i] > ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals