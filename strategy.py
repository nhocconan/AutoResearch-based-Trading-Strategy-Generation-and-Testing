#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_HTFVol
Hypothesis: 12h strategy using Camarilla R1/S1 levels from 1d for breakout entries with 1w EMA50 trend filter and 1d volume spike confirmation.
Enters long when price closes above R1 with 1w uptrend and elevated volume (>1.5x 20-period average).
Enters short when price closes below S1 with 1w downtrend and elevated volume.
Exits on opposite Camarilla level touch (S1/R1) or 1w trend reversal.
Designed for low trade frequency (12-37/year) with discrete position sizing (0.25) to minimize fee drag.
Works in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R1/S1 from 1d OHLC
    # R1 = C + (H-L)*1.1/4, S1 = C - (H-L)*1.1/4
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # 1d volume for confirmation
    vol_1d = df_1d['volume'].values
    
    # 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe (completed 1d bars only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    # Align 1w EMA to 12h timeframe (completed 1w bars only)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current 1d volume > 1.5 * 20-period average
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = vol_1d > (1.5 * vol_avg)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d Camarilla, 1w EMA50, volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1w trend filter and volume confirmation
            # Long: price closes above R1 AND above 1w EMA50 (uptrend)
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S1 AND below 1w EMA50 (downtrend)
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches S1 (opposite level) OR 1w EMA turns bearish (price below EMA)
            if (close_val < s1_val) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R1 (opposite level) OR 1w EMA turns bullish (price above EMA)
            if (close_val > r1_val) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_HTFVol"
timeframe = "12h"
leverage = 1.0