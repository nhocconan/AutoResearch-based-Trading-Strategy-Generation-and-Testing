#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike
Hypothesis: Uses 4h Camarilla R3/S3 levels for breakout entries with 1h volume confirmation and 4h EMA50 trend filter.
Long when price breaks above R3 with volume spike AND 4h close > EMA50 (uptrend).
Short when price breaks below S3 with volume spike AND 4h close < EMA50 (downtrend).
Exit on opposite Camarilla level (R1/S1) or trend reversal.
Designed for 1h timeframe to achieve 60-150 total trades over 4 years (15-37/year) by using 4h for direction and 1h for timing.
Works in both bull and bear markets by following 4h trend while using Camarilla levels for precise entries.
Volume spike filter reduces false signals. Discrete position sizing (0.20) minimizes fee drag.
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
    
    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels: based on previous 4h bar's high, low, close
    # R4 = close + (high - low) * 1.5/2
    # R3 = close + (high - low) * 1.25/2
    # R2 = close + (high - low) * 1.1/2
    # R1 = close + (high - low) * 0.5/2
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 0.5/2
    # S2 = close - (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.25/2
    # S4 = close - (high - low) * 1.5/2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate levels for each 4h bar (using previous bar's values)
    rng = high_4h - low_4h
    r3 = close_4h + rng * 1.25 / 2.0
    s3 = close_4h - rng * 1.25 / 2.0
    r1 = close_4h + rng * 0.5 / 2.0
    s1 = close_4h - rng * 0.5 / 2.0
    
    # Align 4h indicators to 1h timeframe (completed bars only)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital (discrete level)
    
    # Warmup: need 4h EMA50 (50) and volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla breakout with 4h EMA50 trend filter AND volume spike
            # Long: price breaks above R3 AND 4h close > EMA50 (uptrend) AND volume spike
            long_condition = (close_val > r3_val) and (close_val > ema_val) and vol_conf
            # Short: price breaks below S3 AND 4h close < EMA50 (downtrend) AND volume spike
            short_condition = (close_val < s3_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price breaks below R1 (take profit) OR trend breaks
            exit_condition = (close_val < r1_val) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price breaks above S1 (take profit) OR trend breaks
            exit_condition = (close_val > s1_val) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0