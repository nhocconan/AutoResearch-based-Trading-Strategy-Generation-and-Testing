#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h strategy using Camarilla R3/S3 levels from 4h for breakout entries with 4h EMA50 trend filter and volume spike confirmation. 
Enter long when price closes above R3 with 4h uptrend (price > EMA50) and volume > 2.0x 20-period average. 
Enter short when price closes below S3 with 4h downtrend (price < EMA50) and volume confirmation. 
Exit on opposite Camarilla level touch (S3/R3) or 4h trend reversal (price crosses EMA50). 
Uses 4h for signal direction, 1h only for entry timing to control trade frequency (~20-50/year). 
Session filter (08-20 UTC) reduces noise. Discrete position sizing (0.20) minimizes fee drag.
Designed to work in both bull and bear markets by following the 4h trend while using Camarilla levels for precise breakout entries.
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
    open_time = prices['open_time'].values
    
    # Get 4h data for Camarilla levels and EMA trend
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h OHLC for Camarilla levels
    o_4h = df_4h['open'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Camarilla levels: R3/S3 from 4h OHLC
    camarilla_r3 = c_4h + (h_4h - l_4h) * 1.1 / 4
    camarilla_s3 = c_4h - (h_4h - l_4h) * 1.1 / 4
    
    # 4h EMA50 for trend filter
    close_4h_series = pd.Series(c_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe (completed bars only)
    r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital (discrete level)
    
    # Warmup: need 4h EMA50 (50) + volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
            # Long: price closes above R3 AND above EMA50 (4h uptrend)
            long_condition = (close_val > r3_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S3 AND below EMA50 (4h downtrend)
            short_condition = (close_val < s3_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price touches S3 (opposite level) OR 4h EMA50 turns bearish (price below EMA)
            if (close_val < s3_val) or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (opposite level) OR 4h EMA50 turns bullish (price above EMA)
            if (close_val > r3_val) or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0