#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeConfirm
Hypothesis: Daily strategy using Camarilla R1/S1 levels from 1d for breakout entries with 1-week EMA20 trend filter and volume confirmation. 
Price must close above/below R1/S1 to enter long/short, with weekly EMA20 confirming 1w trend alignment. 
Volume > 2.0x 20-period average ensures institutional participation. 
Designed for very low trade frequency (<25/year) with discrete position sizing to minimize fee drag.
Works in both bull and bear markets by following the 1w trend while using tight Camarilla levels for precise entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels (daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R1/S1 from 1d OHLC (tighter than R3/S3 for fewer trades)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA20 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d and 1w indicators to 1d timeframe (completed bars only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d (no lookback) + 1w EMA20 (20) + volume avg (20)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_20_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1w EMA20 trend filter and volume confirmation
            # Long: price closes above R1 AND above EMA20 (uptrend)
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S1 AND below EMA20 (downtrend)
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches S1 (opposite level) OR 1w EMA20 turns bearish (price below EMA)
            if (close_val < s1_val) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R1 (opposite level) OR 1w EMA20 turns bullish (price above EMA)
            if (close_val > r1_val) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0