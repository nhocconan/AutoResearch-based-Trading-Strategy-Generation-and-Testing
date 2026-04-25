#!/usr/bin/env python3
"""
4h Williams Alligator + Daily EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator identifies trend direction and market phases (sleeping/awakening/eating). 
Combined with daily EMA50 for higher timeframe trend alignment and volume spike confirmation, 
this captures institutional participation in trending markets while avoiding choppy conditions. 
Works in bull markets (trend continuation) and bear markets (trend reversals). 
4h timeframe targets 20-50 trades/year to avoid fee drag.
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
    
    # Daily data for Williams Alligator and EMA50 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Williams Alligator: Smoothed Moving Average (SMA) with specific periods
    # Jaw (blue): 13-period SMA, smoothed by 8 bars
    # Teeth (red): 8-period SMA, smoothed by 5 bars  
    # Lips (green): 5-period SMA, smoothed by 3 bars
    close_1d = pd.Series(df_1d['close'].values)
    jaw = close_1d.rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    teeth = close_1d.rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    lips = close_1d.rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    # Align HTF indicators to LTF
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Daily EMA50 trend filter
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for daily data (Alligator + EMA) and volume MA
    start_idx = max(50, 20) + 13  # extra for Alligator smoothing
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Alligator conditions: Mouth open (trending) vs closed (sleeping)
        # Mouth open when lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Alligator aligned + volume spike + daily EMA50 trend alignment
            long_entry = bullish_aligned and vol_spike and (curr_close > ema_50_1d_aligned[i])
            short_entry = bearish_aligned and vol_spike and (curr_close < ema_50_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on Alligator sleeping (mouth closed) or trend change
            if not bullish_aligned or curr_close < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on Alligator sleeping (mouth closed) or trend change
            if not bearish_aligned or curr_close > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0