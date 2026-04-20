#!/usr/bin/env python3
# 4h_12h_Camarilla_R3S3_Breakout_Volume_Filter
# Hypothesis: Breakout above R3 or below S3 daily Camarilla levels on 4h timeframe with volume confirmation and 12h trend filter.
# Uses 12h EMA34 to filter trades in the direction of the higher timeframe trend.
# Target: 20-50 trades per year per symbol to minimize fee drag.
# Works in bull/bear via 12h trend filter - only trade with the 12h trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_R3S3_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === Calculate daily pivot levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    
    # === 12h: EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all daily and 12h levels to 4h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r3_1d_val = r3_1d_aligned[i]
        s3_1d_val = s3_1d_aligned[i]
        ema34_12h_val = ema34_12h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3_1d_val) or np.isnan(s3_1d_val) or np.isnan(ema34_12h_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume confirmation and above 12h EMA34
            if (close_val > r3_1d_val and  # Price broke above R3
                vol_ratio_val > 2.0 and  # Volume confirmation
                close_val > ema34_12h_val):  # Only long in 12h uptrend
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume confirmation and below 12h EMA34
            elif (close_val < s3_1d_val and  # Price broke below S3
                  vol_ratio_val > 2.0 and  # Volume confirmation
                  close_val < ema34_12h_val):  # Only short in 12h downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls back below R3 or shows weakness
            if close_val < r3_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S3 or shows weakness
            if close_val > s3_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals