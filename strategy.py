#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 for higher timeframe trend alignment (reduces whipsaw vs shorter TF)
# Camarilla R3/S3 from prior 4h session provide institutional breakout levels
# Volume confirmation (>1.5x 20 EMA) filters low-participation false breakouts
# Session filter (08-20 UTC) to avoid low-liquidity periods
# Discrete sizing 0.20 limits risk and reduces fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Works in both bull and bear: trend filter adapts to higher timeframe direction.

name = "1h_Camarilla_R3S3_4hEMA50_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend direction
    close_4h = pd.Series(df_4h['close'])
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (completed 4h bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    # Typical price for Camarilla calculation
    typical_4h = (high_4h + low_4h + close_4h_vals) / 3.0
    range_4h = high_4h - low_4h
    
    # Camarilla R3, S3 levels (most significant for breakouts)
    camarilla_r3 = close_4h_vals + 1.1 * range_4h / 2.0
    camarilla_s3 = close_4h_vals - 1.1 * range_4h / 2.0
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar only)
    r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla H4/L4 midpoint OR trend changes OR weak volume
            camarilla_h4 = close_4h_vals + 1.1 * range_4h / 4.0
            camarilla_l4 = close_4h_vals - 1.1 * range_4h / 4.0
            h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
            l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
            midpoint = (h4_aligned[i] + l4_aligned[i]) / 2.0
            
            if (close[i] < midpoint or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla H4/L4 midpoint OR trend changes OR weak volume
            camarilla_h4 = close_4h_vals + 1.1 * range_4h / 4.0
            camarilla_l4 = close_4h_vals - 1.1 * range_4h / 4.0
            h4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h4)
            l4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l4)
            midpoint = (h4_aligned[i] + l4_aligned[i]) / 2.0
            
            if (close[i] > midpoint or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals