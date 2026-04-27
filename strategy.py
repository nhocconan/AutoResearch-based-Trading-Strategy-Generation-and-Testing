#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeConfirm
Hypothesis: Daily strategy using Camarilla R1/S1 levels from weekly OHLC for breakout entries with weekly EMA21 trend filter and volume confirmation. 
Enter long when price closes above weekly R1 with weekly uptrend (price > weekly EMA21) and volume > 1.8x 20-day average. 
Enter short when price closes below weekly S1 with weekly downtrend (price < weekly EMA21) and volume confirmation. 
Exit on opposite Camarilla level touch (S1/R1) or weekly trend reversal (price crosses weekly EMA21). 
Designed for low trade frequency (~10-25/year) with discrete position sizing (0.30) to minimize fee drag and maximize edge.
Works in both bull and bear markets by following the weekly trend while using Camarilla levels for precise daily breakout entries.
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
    
    # Get weekly data for Camarilla levels and EMA trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly OHLC for Camarilla levels
    o_1w = df_1w['open'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Camarilla levels: R1/S1 from weekly OHLC (tighter levels for more precise entries)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_1w + (h_1w - l_1w) * 1.1 / 12
    camarilla_s1 = c_1w - (h_1w - l_1w) * 1.1 / 12
    
    # Weekly EMA21 for trend filter
    close_1w_series = pd.Series(c_1w)
    ema_21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly indicators to daily timeframe (completed bars only)
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume confirmation: current volume > 1.8 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.30   # Position size: 30% of capital (discrete level)
    
    # Warmup: need weekly EMA21 (21) + volume avg (20)
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_21_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with weekly EMA21 trend filter and volume confirmation
            # Long: price closes above R1 AND above weekly EMA21 (weekly uptrend)
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S1 AND below weekly EMA21 (weekly downtrend)
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches S1 (opposite level) OR weekly EMA21 turns bearish (price below EMA)
            if (close_val < s1_val) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R1 (opposite level) OR weekly EMA21 turns bullish (price above EMA)
            if (close_val > r1_val) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0