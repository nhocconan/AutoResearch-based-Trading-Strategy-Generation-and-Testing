#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Long when price > Alligator Jaw (13-period smoothed median) and 1d EMA34 up-trend, volume > 1.5x average
# Short when price < Alligator Jaw and 1d EMA34 down-trend, volume > 1.5x average
# Exit when price crosses Alligator Teeth (8-period smoothed median)
# Uses discrete position sizing (0.25) to balance capture and risk.
# Target: 80-180 total trades over 4 years (20-45/year) to avoid fee drag.
# Uses 1d for trend direction, 4h only for entry/exit timing.

name = "4h_Williams_Alligator_1dEMA34_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 4h data
    # Jaw (13-period smoothed median): SMMA(median, 13)
    median_4h = (df_4h['high'].values + df_4h['low'].values) / 2
    jaw_4h = pd.Series(median_4h).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    # Teeth (8-period smoothed median): SMMA(median, 8)
    teeth_4h = pd.Series(median_4h).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    # Lips (5-period smoothed median): SMMA(median, 5)
    lips_4h = pd.Series(median_4h).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Align Alligator components to 4h timeframe (already aligned, but keep for consistency)
    jaw_4h_aligned = align_htf_to_ltf(prices, df_4h, jaw_4h)
    teeth_4h_aligned = align_htf_to_ltf(prices, df_4h, teeth_4h)
    lips_4h_aligned = align_htf_to_ltf(prices, df_4h, lips_4h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation (on 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Volume MA and Alligator Jaw warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_4h_aligned[i]) or np.isnan(teeth_4h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_jaw = jaw_4h_aligned[i]
        curr_teeth = teeth_4h_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Alligator Teeth
            if curr_close < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Alligator Teeth
            if curr_close > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price > Alligator Jaw, 1d EMA34 up-trend, volume confirmed
            if curr_close > curr_jaw and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price < Alligator Jaw, 1d EMA34 down-trend, volume confirmed
            elif curr_close < curr_jaw and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals