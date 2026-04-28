#!/usr/bin/env python3
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
    
    # Get 1d data for Williams Alligator and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Alligator lines on 1d: Jaw(13,8), Teeth(8,5), Lips(5,3)
    # Smoothed with SMMA (using EWMA as approximation with proper adjust)
    jaw_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_1d = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_1d = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align Alligator lines to 6h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Trend filter: price > Teeth = uptrend, price < Teeth = downtrend
    uptrend = close > teeth_aligned
    downtrend = close < teeth_aligned
    
    # Alligator sleeping condition: Jaw, Teeth, Lips intertwined (market ranging)
    # Alligator awake: lines separated (trending)
    # We trade only when Alligator is awake (trending)
    jaw_above_teeth = jaw_aligned > teeth_aligned
    teeth_above_lips = teeth_aligned > lips_aligned
    jaw_below_teeth = jaw_aligned < teeth_aligned
    teeth_below_lips = teeth_aligned < lips_aligned
    
    # Alligator awake: either all lines up (jaw>teeth>lips) or all down (jaw<teeth<lips)
    alligator_awake_up = jaw_above_teeth & teeth_above_lips
    alligator_awake_down = jaw_below_teeth & teeth_below_lips
    alligator_awake = alligator_awake_up | alligator_awake_down
    
    # Volume confirmation: current volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma_20
    
    # Entry conditions: Alligator awake + price extreme relative to Lips/Jaw
    # Long: price crosses above Lips in uptrend
    # Short: price crosses below Jaw in downtrend
    long_entry = alligator_awake & uptrend & (close > lips_aligned) & volume_filter
    short_entry = alligator_awake & downtrend & (close < jaw_aligned) & volume_filter
    
    # Exit conditions: price returns to Teeth (neutral zone) or Alligator starts sleeping
    long_exit = (close < teeth_aligned) | (~alligator_awake)
    short_exit = (close > teeth_aligned) | (~alligator_awake)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Handle entries and exits
        if long_entry[i] and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry[i] and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and long_exit[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and short_exit[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsAlligator_Awake_Trend_Volume"
timeframe = "6h"
leverage = 1.0