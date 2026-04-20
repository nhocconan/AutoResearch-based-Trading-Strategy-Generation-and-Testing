#!/usr/bin/env python3
# 6h_1w_1d_Pivot_R3S3_Breakout_Volume_TrendFilter
# Hypothesis: Breakout above weekly R3 or below weekly S3 with volume confirmation and daily trend filter on 6h timeframe.
# Uses weekly pivot levels for major support/resistance, EMA34 from daily to filter trend direction, and volume spike for confirmation.
# Works in bull/bear via EMA34 filter - only trade breakouts in direction of daily trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Pivot_R3S3_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate weekly pivot levels (R3, S3) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    
    # === Calculate EMA34 on daily close for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all weekly data to 6h
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Align all daily data to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r3_1w_val = r3_1w_aligned[i]
        s3_1w_val = s3_1w_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3_1w_val) or np.isnan(s3_1w_val) or 
            np.isnan(ema_34_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R3 with volume confirmation and uptrend (price > EMA34)
            if (close_val > r3_1w_val and  # Price broke above R3
                ema_34_val > 0 and  # Valid EMA
                close_val > ema_34_val and  # Uptrend filter: price above EMA34
                vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S3 with volume confirmation and downtrend (price < EMA34)
            elif (close_val < s3_1w_val and  # Price broke below S3
                  ema_34_val > 0 and  # Valid EMA
                  close_val < ema_34_val and  # Downtrend filter: price below EMA34
                  vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below EMA34 or breaks below S3 (invalidates uptrend)
            if close_val < ema_34_val or close_val < s3_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above EMA34 or breaks above R3 (invalidates downtrend)
            if close_val > ema_34_val or close_val > r3_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals