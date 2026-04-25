#!/usr/bin/env python3
"""
1d Camarilla Pivot R3/S3 Breakout + 1w EMA34 Trend + Volume Spike Confirmation
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance on daily timeframe.
Breakouts above R3 or below S3 with volume confirmation and 1w EMA34 trend filter capture
strong momentum moves. Works in both bull and bear markets by taking breakouts in direction
of higher timeframe trend. Designed for ~50-80 trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need 34 for EMA34
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    R3 = pivot + (high_1d - low_1d) * 1.1 / 4.0
    S3 = pivot - (high_1d - low_1d) * 1.1 / 4.0
    R2 = pivot + (high_1d - low_1d) * 1.1 / 6.0
    S2 = pivot - (high_1d - low_1d) * 1.1 / 6.0
    R1 = pivot + (high_1d - low_1d) * 1.1 / 12.0
    S1 = pivot - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 1d timeframe (already 1d, but using helper for consistency)
    R3_1d = align_htf_to_ltf(prices, df_1d, R3)
    S3_1d = align_htf_to_ltf(prices, df_1d, S3)
    R2_1d = align_htf_to_ltf(prices, df_1d, R2)
    S2_1d = align_htf_to_ltf(prices, df_1d, S2)
    R1_1d = align_htf_to_ltf(prices, df_1d, R1)
    S1_1d = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for additional trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 34:
        close_1w = pd.Series(df_1w['close'])
        ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
        ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    else:
        ema_34_1w_aligned = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume spike confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_1d[i]) or np.isnan(S3_1d[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        r3 = R3_1d[i]
        s3 = S3_1d[i]
        ema_34_val = ema_34_1d_aligned[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        # Trend filter: price above/below 1d EMA34 AND 1w EMA34 (if available)
        if not np.isnan(ema_34_1w_val):
            trend_filter = (curr_close > ema_34_val and curr_close > ema_34_1w_val) or \
                          (curr_close < ema_34_val and curr_close < ema_34_1w_val)
        else:
            trend_filter = curr_close > ema_34_val or curr_close < ema_34_val
        
        if position == 0:
            # Look for breakout above R3 or below S3 with volume and trend confirmation
            long_breakout = curr_close > r3 and volume_confirm and trend_filter
            short_breakout = curr_close < s3 and volume_confirm and trend_filter
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below R2 or R1 (profit taking or reversal)
            if curr_close < r2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above S2 or S1 (profit taking or reversal)
            if curr_close > s2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0