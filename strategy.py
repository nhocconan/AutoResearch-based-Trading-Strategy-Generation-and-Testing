#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA34 Trend + Volume Spike
# Long when price > Alligator Jaw (13-period smoothed median) AND price > Alligator Teeth (8-period smoothed median) 
# AND price > Alligator Lips (5-period smoothed median) AND 1d EMA34 up AND volume > 2.0x 20-bar avg
# Short when price < Alligator Jaw AND price < Alligator Teeth AND price < Alligator Lips AND 1d EMA34 down AND volume > 2.0x 20-bar avg
# Exit when price crosses back below Alligator Teeth (for long) or above Alligator Teeth (for short)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 20-50 trades/year on 4h timeframe.
# Williams Alligator identifies trend presence and direction, 1d EMA34 filters higher timeframe trend,
# volume confirmation ensures breakout strength. Works in both bull (trend following) and bear (counter-trend reversals) markets.

name = "4h_WilliamsAlligator_1dEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 4h data (using median price)
    median_price = (high + low) / 2.0
    
    # Alligator Jaw (13-period, smoothed by 8 bars)
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Alligator Teeth (8-period, smoothed by 5 bars)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Alligator Lips (5-period, smoothed by 3 bars)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 8, 5, 34)  # volume MA, Alligator components, EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema34 = ema_34_1d_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses back below Alligator Teeth
            if curr_close < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above Alligator Teeth
            if curr_close > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price > Alligator Jaw > Teeth > Lips AND 1d EMA34 up AND volume confirmation
            if (curr_close > curr_jaw and curr_jaw > curr_teeth and curr_teeth > curr_lips and 
                curr_ema34 > ema_34_1d_aligned[i-1] and vol_conf):
                signals[i] = 0.25
                position = 1
            # Short when price < Alligator Jaw < Teeth < Lips AND 1d EMA34 down AND volume confirmation
            elif (curr_close < curr_jaw and curr_jaw < curr_teeth and curr_teeth < curr_lips and 
                  curr_ema34 < ema_34_1d_aligned[i-1] and vol_conf):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals