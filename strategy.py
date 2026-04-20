#!/usr/bin/env python3
# 4h_1d_Camarilla_R1S1_Breakout_Volume_KeltnerFilter_v1
# Hypothesis: Breakouts of daily R1/S1 with volume confirmation (3.0x) and price outside Keltner Channel (2*ATR)
# to avoid false breakouts in ranging markets. Exit when price returns to midpoint of daily range.
# Designed to work in both bull and bear markets by using volatility-based filtering.
# Target: 20-40 trades per year per symbol to avoid excessive fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1S1_Breakout_Volume_KeltnerFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily Camarilla pivots ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    # Midpoint for exit: (high + low) / 2
    midpoint_1d = (high_1d + low_1d) / 2.0
    
    # === Calculate daily ATR for Keltner Channel ===
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Keltner Channel: Upper = EMA(close) + 2*ATR, Lower = EMA(close) - 2*ATR
    ema_close_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper_1d = ema_close_1d + 2.0 * atr_1d
    keltner_lower_1d = ema_close_1d - 2.0 * atr_1d
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA and EMA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        midpoint_val = midpoint_aligned[i]
        keltner_upper_val = keltner_upper_aligned[i]
        keltner_lower_val = keltner_lower_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(midpoint_val) or 
            np.isnan(keltner_upper_val) or np.isnan(keltner_lower_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation AND above Keltner Upper
            if close_val > r1_val and vol_ratio_val > 3.0 and close_val > keltner_upper_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation AND below Keltner Lower
            elif close_val < s1_val and vol_ratio_val > 3.0 and close_val < keltner_lower_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below daily midpoint
            if close_val <= midpoint_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above daily midpoint
            if close_val >= midpoint_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals