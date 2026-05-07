#!/usr/bin/env python3
name = "6h_12h_1d_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h trend: EMA(20) on 12h close
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate daily trend: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    range_hl_1d = prev_high_1d - prev_low_1d
    
    # Camarilla R3/S3 levels
    s3_1d = prev_close_1d - (range_hl_1d * 1.26 / 4)
    r3_1d = prev_close_1d + (range_hl_1d * 1.26 / 4)
    
    # Align daily levels to 6h timeframe
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S3 with volume and both 12h/1d uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend_12h = ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]
            uptrend_1d = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s3_1d_aligned[i] and vol_condition and uptrend_12h and uptrend_1d:
                signals[i] = 0.25
                position = 1
            # Short: price below R3 with volume and both 12h/1d downtrend
            elif close[i] < r3_1d_aligned[i] and vol_condition and not uptrend_12h and not uptrend_1d:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S3 or trend breaks
            if close[i] < s3_1d_aligned[i] or not (ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R3 or trend breaks
            if close[i] > r3_1d_aligned[i] or not (ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h/1d trend and volume confirmation
# - Daily Camarilla R3/S3 act as strong support/resistance levels
# - Breakout above S3 with volume in both 12h and 1d uptrend = long opportunity
# - Breakdown below R3 with volume in both 12h and 1d downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Requires alignment of both 12h and 1d trends for higher conviction
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to S3/R3 or trend breaks on either timeframe
# - Position size 0.25 targets ~50-150 trades over 4 years, avoiding fee drag
# - Uses actual daily Camarilla levels (not weekly) for better responsiveness
# - Combines multiple timeframes to reduce false signals while maintaining trend sensitivity