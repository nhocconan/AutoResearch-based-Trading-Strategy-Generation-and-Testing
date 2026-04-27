#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v2
Hypothesis: Refined version of prior strategy with tighter entry conditions to reduce trade frequency and avoid overtrading. Uses Camarilla R3/S3 from 1d for breakout entries, confirmed by 1d EMA34 trend and volume spike (>2.5x 20-period average). Adds 2-bar hold minimum to prevent whipsaw. Designed for 15-25 trades/year to minimize fee drag while maintaining edge in both bull and bear markets via trend-following breakouts from institutional pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (wait for previous day's close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend confirmation
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.5 * 20-period average (stricter)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Track bars since last entry to enforce minimum hold
    bars_since_entry = 0
    
    # Warmup: need enough data for EMA and volume
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Only allow entry after minimum 2 bars since last exit
            if bars_since_entry >= 2:
                # Long: price breaks above R3, above EMA34 trend, volume confirmation
                if close[i] > camarilla_r3_val and close[i] > ema34_val and vol_conf:
                    signals[i] = size
                    position = 1
                    bars_since_entry = 0
                # Short: price breaks below S3, below EMA34 trend, volume confirmation
                elif close[i] < camarilla_s3_val and close[i] < ema34_val and vol_conf:
                    signals[i] = -size
                    position = -1
                    bars_since_entry = 0
            else:
                bars_since_entry += 1
        elif position == 1:
            # Exit long: price crosses below EMA34
            if close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = size
                bars_since_entry += 1
        elif position == -1:
            # Exit short: price crosses above EMA34
            if close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -size
                bars_since_entry += 1
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0