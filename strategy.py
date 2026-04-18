#!/usr/bin/env python3
"""
1d Williams Alligator + Volume Spike + Weekly Trend Filter
Strategy: Use Williams Alligator (3 SMAs) to detect trend direction.
          Enter long when price > Alligator Jaw with volume spike and weekly EMA20 uptrend.
          Enter short when price < Alligator Jaw with volume spike and weekly EMA20 downtrend.
          Williams Alligator helps avoid whipsaws in ranging markets.
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    ema_20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Williams Alligator components (13,8,5 smoothed by 8,5,3)
    # Jaw (13-period SMMA smoothed by 8)
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    # Teeth (8-period SMMA smoothed by 5)
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    # Lips (5-period SMMA smoothed by 3)
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Align Alligator components to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # Already daily data
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw_aligned[i]
        ema_20w = ema_20_1w_aligned[i]
        
        # Alligator alignment: jaw > teeth > lips = uptrend, reverse = downtrend
        # But we simplify to price vs jaw for entry
        
        if position == 0:
            # Long: price above jaw with volume spike and weekly uptrend
            if (price > jaw_val and volume_spike[i] and price > ema_20w):
                signals[i] = 0.25
                position = 1
            # Short: price below jaw with volume spike and weekly downtrend
            elif (price < jaw_val and volume_spike[i] and price < ema_20w):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price crosses below jaw or weekly trend turns down
            if price < jaw_val or price < ema_20w:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price crosses above jaw or weekly trend turns up
            if price > jaw_val or price > ema_20w:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WilliamsAlligator_VolumeSpike_WeeklyEMA20"
timeframe = "1d"
leverage = 1.0