#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for higher timeframe trend alignment (stable in both bull/bear markets)
# Camarilla R3/S3 from prior 1d session provide institutional breakout levels
# Volume confirmation (>1.5x 20 EMA) filters low-participation false breakouts
# Session filter (08-20 UTC) reduces noise trades. Discrete sizing 0.20 limits risk.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Works in both bull and bear: trend filter adapts to higher timeframe direction.

name = "1h_Camarilla_R3S3_1dEMA34_VolumeSpike_Session"
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 1h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_1d = (high_1d + low_1d + close_1d_vals) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla R3, S3 levels (most significant for breakouts)
    camarilla_r3 = close_1d_vals + 1.1 * range_1d / 2.0
    camarilla_s3 = close_1d_vals - 1.1 * range_1d / 2.0
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla H4/L4 midpoint OR trend changes OR weak volume
            camarilla_h4 = close_1d_vals + 1.1 * range_1d / 4.0
            camarilla_l4 = close_1d_vals - 1.1 * range_1d / 4.0
            h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
            l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
            midpoint = (h4_aligned[i] + l4_aligned[i]) / 2.0
            
            if (close[i] < midpoint or 
                close[i] < ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla H4/L4 midpoint OR trend changes OR weak volume
            camarilla_h4 = close_1d_vals + 1.1 * range_1d / 4.0
            camarilla_l4 = close_1d_vals - 1.1 * range_1d / 4.0
            h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
            l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
            midpoint = (h4_aligned[i] + l4_aligned[i]) / 2.0
            
            if (close[i] > midpoint or 
                close[i] > ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals