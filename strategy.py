#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 4h Camarilla pivot levels for precise entry/exit zones
# 4h EMA50 provides strong trend filter to avoid counter-trend trades in ranging markets
# Volume > 2.0x average confirms institutional participation and reduces false breakouts
# Session filter (08-20 UTC) reduces noise trades
# Discrete position sizing (0.20) with Camarilla H3/L3 exit for quick profit taking
# Designed for ~15-37 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via trend filter - only trades in direction of 4h EMA50

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pivot point
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    # Range
    range_4h = high_4h - low_4h
    
    # Camarilla levels
    r3_4h = close_4h + range_4h * 1.1 / 4.0
    s3_4h = close_4h - range_4h * 1.1 / 4.0
    r4_4h = close_4h + range_4h * 1.1 / 2.0
    s4_4h = close_4h - range_4h * 1.1 / 2.0
    h3_4h = close_4h + range_4h * 1.1 / 6.0
    l3_4h = close_4h - range_4h * 1.1 / 6.0
    
    # Align 4h Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume MA and 4h EMA50 warmup
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or 
            np.isnan(r4_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_r3 = r3_4h_aligned[i]
        curr_s3 = s3_4h_aligned[i]
        curr_h3 = h3_4h_aligned[i]
        curr_l3 = l3_4h_aligned[i]
        curr_r4 = r4_4h_aligned[i]
        curr_s4 = s4_4h_aligned[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Camarilla L3 (profit taking at support)
            if curr_close < curr_l3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla H3 (profit taking at resistance)
            if curr_close > curr_h3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 2.0x 20-period average
            vol_spike = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above R3 with 4h EMA50 uptrend and volume spike
            if curr_high > curr_r3 and curr_close > curr_ema50_4h and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3 with 4h EMA50 downtrend and volume spike
            elif curr_low < curr_s3 and curr_close < curr_ema50_4h and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals