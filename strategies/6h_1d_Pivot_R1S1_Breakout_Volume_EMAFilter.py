#!/usr/bin/env python3
# 6h_1d_Pivot_R1S1_Breakout_Volume_EMAFilter
# Hypothesis: Breakout above R1 or below S1 daily pivot levels with volume confirmation and EMA34 trend filter on 6h timeframe.
# Uses 1d pivot levels for key support/resistance, EMA34 to filter trend direction, and volume spike for confirmation.
# Works in bull/bear via EMA34 filter - only trade breakouts in direction of 6h trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R1S1_Breakout_Volume_EMAFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
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
    
    # === Calculate EMA34 on 1d close for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 1d data to 6h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r1_1d_val = r1_1d_aligned[i]
        s1_1d_val = s1_1d_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_1d_val) or np.isnan(s1_1d_val) or 
            np.isnan(ema_34_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R1 with volume confirmation and uptrend (price > EMA34)
            if (close_val > r1_1d_val and  # Price broke above R1
                ema_34_val > 0 and  # Valid EMA
                close_val > ema_34_val and  # Uptrend filter: price above EMA34
                vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 with volume confirmation and downtrend (price < EMA34)
            elif (close_val < s1_1d_val and  # Price broke below S1
                  ema_34_val > 0 and  # Valid EMA
                  close_val < ema_34_val and  # Downtrend filter: price below EMA34
                  vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below EMA34 or breaks below S1 (invalidates uptrend)
            if close_val < ema_34_val or close_val < s1_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above EMA34 or breaks above R1 (invalidates downtrend)
            if close_val > ema_34_val or close_val > r1_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals