#!/usr/bin/env python3
# 4h_1d_Camarilla_R1S1_MeanReversion_Volume
# Hypothesis: Mean reversion at 1d Camarilla R1/S1 levels with volume confirmation on 4h timeframe.
# Uses 1d trend filter to avoid counter-trend trades. Works in bull/bear via 1d trend filter - only trade against the 1d trend at reversal points.
# Target: 100-180 trades over 4 years (25-45/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1S1_MeanReversion_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for pivot levels and trend
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
    
    # === 1d trend filter: EMA(34) for trend direction ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all 1d levels and trend to 4h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r1_1d_val = r1_1d_aligned[i]
        s1_1d_val = s1_1d_aligned[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_1d_val) or np.isnan(s1_1d_val) or 
            np.isnan(ema_34_1d_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price rejects S1 (bounces off support) with volume confirmation and against 1d trend
            if (close_val < s1_1d_val and  # Price touched or went below S1
                prices['low'].iloc[i] <= s1_1d_val and  # Confirmed touch of S1
                close_val > s1_1d_val and  # Now bouncing back above S1
                vol_ratio_val > 2.0 and  # Volume confirmation
                close_val < ema_34_1d_val):  # Below 1d EMA (downtrend) - mean reversion long
                signals[i] = 0.25
                position = 1
            # Short: Price rejects R1 (bounces off resistance) with volume confirmation and against 1d trend
            elif (close_val > r1_1d_val and  # Price touched or went above R1
                  prices['high'].iloc[i] >= r1_1d_val and  # Confirmed touch of R1
                  close_val < r1_1d_val and  # Now falling back below R1
                  vol_ratio_val > 2.0 and  # Volume confirmation
                  close_val > ema_34_1d_val):  # Above 1d EMA (uptrend) - mean reversion short
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price reaches R1 or shows weakness
            if close_val >= r1_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price reaches S1 or shows weakness
            if close_val <= s1_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals