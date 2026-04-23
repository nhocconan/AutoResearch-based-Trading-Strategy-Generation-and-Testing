#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
- Long when price breaks above 4h Camarilla R3 AND price > 12h EMA50 AND volume > 2.0x 20-period average
- Short when price breaks below 4h Camarilla S3 AND price < 12h EMA50 AND volume > 2.0x 20-period average
- Exit when price crosses the 4h Camarilla pivot point (mean reversion to median)
- Uses 12h EMA50 for HTF trend alignment to avoid counter-trend trades and capture major trend
- Volume spike ensures institutional participation and reduces false breakouts
- Uses 4h primary timeframe with 12h HTF for signal direction to balance trade frequency
- Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag
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
    
    # Get 12h data for EMA50 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 4h data for Camarilla levels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    pivot_4h = pd.Series(typical_price_4h).rolling(window=20, min_periods=20).mean().values
    range_hl_4h = pd.Series(df_4h['high'] - df_4h['low']).rolling(window=20, min_periods=20).mean().values
    camarilla_r3_4h = pivot_4h + range_hl_4h * 1.1 / 4.0
    camarilla_s3_4h = pivot_4h - range_hl_4h * 1.1 / 4.0
    camarilla_pivot_4h = pivot_4h  # Camarilla pivot point
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    camarilla_pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot_4h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 51, 21)  # Need 20 for Camarilla, 51 for EMA50 (50+1), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_4h_aligned[i]) or 
            np.isnan(camarilla_s3_4h_aligned[i]) or 
            np.isnan(camarilla_pivot_4h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions (using 4h Camarilla levels)
        breakout_up = close[i] > camarilla_r3_4h_aligned[i]  # Break above Camarilla R3
        breakout_down = close[i] < camarilla_s3_4h_aligned[i]  # Break below Camarilla S3
        
        # Trend filter (using 12h EMA50)
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses 4h Camarilla pivot point (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below pivot
                if close[i] < camarilla_pivot_4h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above pivot
                if close[i] > camarilla_pivot_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0