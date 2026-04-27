#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts on 6h with 1d EMA34 trend filter and volume spike confirmation. Uses discrete position sizing (0.25) to reduce fee drag. Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year). Works in both bull and bear markets by following 1d trend direction while using Camarilla levels for precise entries. Volume spike filter ensures momentum confirmation. Avoids overtrading by using tighter R3/S3 levels (vs R1/S1) and stricter volume confirmation.
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
    
    # Get 1d data for EMA trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R3/S3 from 1d OHLC (wider than R1/S1 for fewer false breakouts)
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_s3 = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # Align 1d indicators to 6h timeframe (completed bars only)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.8 * 30-period average (stricter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    
    # Warmup: need 1d EMA34 (34) + volume avg (30)
    start_idx = max(34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
            # Long: price closes above R3 AND above EMA34 (1d uptrend) AND volume spike
            long_condition = (close_val > r3_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S3 AND below EMA34 (1d downtrend) AND volume spike
            short_condition = (close_val < s3_val) and (close_val < ema_val) and vol_conf
            
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
            # 1. Price touches S3 (opposite Camarilla level)
            # 2. 1d EMA34 turns bearish (price below EMA)
            exit_condition = (close_val < s3_val) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions:
            # 1. Price touches R3 (opposite Camarilla level)
            # 2. 1d EMA34 turns bullish (price above EMA)
            exit_condition = (close_val > r3_val) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0