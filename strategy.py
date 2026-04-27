#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: Daily Camarilla R1/S1 breakouts with 1-week EMA34 trend filter and volume spike confirmation. Designed for 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year). Uses discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear markets by following 1-week trend direction while using Camarilla levels for precise entries. Volume confirmation filters out low-quality breakouts.
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R1/S1 from 1d OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align 1d indicators to 1d timeframe (no alignment needed for same TF, but keep for consistency)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    
    # Warmup: need 1w EMA34 (34) + volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_1w_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1w EMA34 trend filter and volume spike
            # Long: price closes above R1 AND above EMA34 (1w uptrend) AND volume spike
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S1 AND below EMA34 (1w downtrend) AND volume spike
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions:
            # 1. Price touches S1 (opposite Camarilla level)
            # 2. 1w EMA34 turns bearish (price below EMA)
            exit_condition = (close_val < s1_val) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions:
            # 1. Price touches R1 (opposite Camarilla level)
            # 2. 1w EMA34 turns bullish (price above EMA)
            exit_condition = (close_val > r1_val) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0