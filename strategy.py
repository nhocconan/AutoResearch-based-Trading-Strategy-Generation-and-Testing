#!/usr/bin/env python3
"""
12h Williams Alligator Breakout with 1d EMA Trend Filter and Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence.
Breakouts above Lips or below Jaw with 1d EMA trend alignment and volume spikes
capture strong moves in both bull and bear markets. Uses 12h timeframe with 1d HTF
for trend filter. Targets 50-150 trades over 4 years (12-37/year) to avoid fee drag.
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
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h median price
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Alligator lines: SMAs of median price
    # Jaw: 13-period SMA, shifted 8 bars ahead
    # Teeth: 8-period SMA, shifted 5 bars ahead  
    # Lips: 5-period SMA, shifted 3 bars ahead
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d for volume confirmation
    vol_ma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_20_1d[i] = np.mean(df_1d['volume'].values[i-19:i+1])
    
    # Align 1d indicators to 12h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 20-period volume MA for 12h volume spike
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Alligator and volume MA
    start_idx = max(20, 13)  # 20 for volume MA, 13 for Alligator jaw
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(lips[i]) or np.isnan(jaw[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vol_ma_20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        lips_val = lips[i]
        jaw_val = jaw[i]
        vol_ma_1d = vol_ma_20_1d_aligned[i]
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 1.8 * 20-period 12h average
        volume_confirm = curr_volume > 1.8 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: price crosses above Lips, above 1d EMA, volume confirmation
            long_entry = (curr_close > lips_val and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price crosses below Jaw, below 1d EMA, volume confirmation
            short_entry = (curr_close < jaw_val and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Teeth OR below 1d EMA
            if curr_close < teeth[i] or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Teeth OR above 1d EMA
            if curr_close > teeth[i] or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0