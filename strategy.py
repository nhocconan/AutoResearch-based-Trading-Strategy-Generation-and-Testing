#!/usr/bin/env python3
# 4h_1d_4w_Pivot_R1S1_Breakout_VolumeTrend
# Hypothesis: Breakout above 1d R1 or below S1 pivot levels on 4h timeframe, with volume confirmation and 4w EMA trend filter.
# Uses daily pivot levels for key support/resistance, EMA50 on 4w to filter long-term trend, and volume spike for confirmation.
# Works in bull/bear via EMA50 filter - only trade breakouts in direction of 4w trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_4w_Pivot_R1S1_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4w data ONCE before loop for EMA trend filter
    df_4w = get_htf_data(prices, '4w')
    if len(df_4w) < 2:
        return np.zeros(n)
    
    # === Calculate 1d pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # === Calculate EMA50 on 4w close for trend filter ===
    ema_50_4w = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all HTF data to 4h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_50_4w_aligned = align_htf_to_ltf(prices, df_4w, ema_50_4w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r1_1d_val = r1_1d_aligned[i]
        s1_1d_val = s1_1d_aligned[i]
        ema_50_4w_val = ema_50_4w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_1d_val) or np.isnan(s1_1d_val) or 
            np.isnan(ema_50_4w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R1 with volume confirmation and uptrend (price > EMA50_4w)
            if (close_val > r1_1d_val and  # Price broke above R1
                ema_50_4w_val > 0 and  # Valid EMA
                close_val > ema_50_4w_val and  # Uptrend filter: price above EMA50_4w
                vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 with volume confirmation and downtrend (price < EMA50_4w)
            elif (close_val < s1_1d_val and  # Price broke below S1
                  ema_50_4w_val > 0 and  # Valid EMA
                  close_val < ema_50_4w_val and  # Downtrend filter: price below EMA50_4w
                  vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below EMA50_4w or breaks below S1 (invalidates uptrend)
            if close_val < ema_50_4w_val or close_val < s1_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above EMA50_4w or breaks above R1 (invalidates downtrend)
            if close_val > ema_50_4w_val or close_val > r1_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals