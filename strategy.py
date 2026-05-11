#!/usr/bin/env python3
"""
12h_1d_Camarilla_R3_S3_Breakout_TrendFilter_Volume
Hypothesis: Uses daily Camarilla pivot levels (R3/S3) from 1d timeframe for breakout entries on 12h.
Trend filter: 1d EMA34 (price above for long, below for short). Volume confirmation required.
Designed to work in both bull and bear markets by following higher-timeframe trend and using
symmetrical breakout levels. Targets low trade frequency (15-30/year) via strict entry conditions.
"""

name = "12h_1d_Camarilla_R3_S3_Breakout_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    # Pivot point
    pivot = (high + low + close) / 3
    range_ = high - low
    
    # Resistance levels
    r1 = close + (range_ * 1.1 / 12)
    r2 = close + (range_ * 1.1 / 6)
    r3 = close + (range_ * 1.1 / 4)
    r4 = close + (range_ * 1.1 / 2)
    
    # Support levels
    s1 = close - (range_ * 1.1 / 12)
    s2 = close - (range_ * 1.1 / 6)
    s3 = close - (range_ * 1.1 / 4)
    s4 = close - (range_ * 1.1 / 2)
    
    return r3, s3  # Return only R3 and S3 for breakout

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla for Breakout Levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    r3_1d, s3_1d = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align daily Camarilla levels to 12h timeframe
    r3_1d_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # --- Daily EMA34 for Trend Filter ---
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection (4-period average = 2 days) ---
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_12h[i]) or np.isnan(s3_1d_12h[i]) or 
            np.isnan(ema_34_1d_12h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R3 + above EMA34 + volume
            if (close[i] > r3_1d_12h[i] and 
                close[i] > ema_34_1d_12h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below EMA34 + volume
            elif (close[i] < s3_1d_12h[i] and 
                  close[i] < ema_34_1d_12h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to EMA34 or trend flip
            if position == 1:
                # Exit long: price crosses below EMA34
                if close[i] < ema_34_1d_12h[i] and close[i-1] >= ema_34_1d_12h[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above EMA34
                if close[i] > ema_34_1d_12h[i] and close[i-1] <= ema_34_1d_12h[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals