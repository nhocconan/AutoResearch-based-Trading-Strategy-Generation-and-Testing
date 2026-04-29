#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels from 1d HTF for precise entry/exit zones
# 1d EMA34 provides strong trend filter to avoid counter-trend trades in ranging markets
# Volume > 1.5x average confirms participation and reduces false breakouts
# Discrete position sizing (0.25) with Camarilla H1/L1 exit for quick profit taking
# Designed for ~12-30 trades/year to minimize fee drag while capturing strong moves
# Works in bull/bear via trend filter - only trades in direction of 1d EMA34

name = "12h_Camarilla_R1S1_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels (R1/S1 for entry, H1/L1 for exit)
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    h1_1d = close_1d + range_1d * 1.1 / 6.0
    l1_1d = close_1d - range_1d * 1.1 / 6.0
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d Camarilla levels and EMA to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    h1_1d_aligned = align_htf_to_ltf(prices, df_1d, h1_1d)
    l1_1d_aligned = align_htf_to_ltf(prices, df_1d, l1_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Volume MA and 1d EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(h1_1d_aligned[i]) or np.isnan(l1_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_r1 = r1_1d_aligned[i]
        curr_s1 = s1_1d_aligned[i]
        curr_h1 = h1_1d_aligned[i]
        curr_l1 = l1_1d_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Camarilla L1 (profit taking at support)
            if curr_close < curr_l1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla H1 (profit taking at resistance)
            if curr_close > curr_h1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume spike confirmation: current volume > 1.5x 20-period average
            vol_spike = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above R1 with 1d EMA34 uptrend and volume spike
            if curr_high > curr_r1 and curr_close > curr_ema34_1d and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with 1d EMA34 downtrend and volume spike
            elif curr_low < curr_s1 and curr_close < curr_ema34_1d and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals