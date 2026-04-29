#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 Trend Filter and Volume Spike
# Long when price > Alligator Jaw (13-period SMMA) AND Jaw > Teeth (8-period SMMA) AND Teeth > Lips (5-period SMMA) 
# AND price > 1d EMA50 AND volume > 1.5x 20-bar avg volume
# Short when price < Alligator Jaw AND Jaw < Teeth AND Teeth < Lips AND price < 1d EMA50 AND volume > 1.5x 20-bar avg volume
# Exit when Alligator lines crossover (Jaw-Teeth or Teeth-Lips) or price crosses opposite Alligator line
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.
# Williams Alligator identifies trend phases (sleeping, awakening, eating). 1d EMA50 filters counter-trend moves,
# volume confirmation ensures trend strength. Works in bull via trend continuation, in bear via trend continuation.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilders"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    if len(source) < length:
        return result
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (Prev SMMA * (length-1) + Current Price) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA of median price
    median_price = (high + low) / 2.0
    jaw = smma(median_price, 13)  # Alligator's Jaw (13-period SMMA)
    teeth = smma(median_price, 8)  # Alligator's Teeth (8-period SMMA)
    lips = smma(median_price, 5)   # Alligator's Lips (5-period SMMA)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator lines crossover (Jaw-Teeth or Teeth-Lips) or price closes below Teeth
            jaw_teeth_cross = (jaw[i] <= teeth[i] and jaw[i-1] > teeth[i-1]) or \
                             (jaw[i] >= teeth[i] and jaw[i-1] < teeth[i-1])
            teeth_lips_cross = (teeth[i] <= lips[i] and teeth[i-1] > lips[i-1]) or \
                              (teeth[i] >= lips[i] and teeth[i-1] < lips[i-1])
            price_below_teeth = curr_close < teeth[i]
            
            if jaw_teeth_cross or teeth_lips_cross or price_below_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines crossover (Jaw-Teeth or Teeth-Lips) or price closes above Teeth
            jaw_teeth_cross = (jaw[i] <= teeth[i] and jaw[i-1] > teeth[i-1]) or \
                             (jaw[i] >= teeth[i] and jaw[i-1] < teeth[i-1])
            teeth_lips_cross = (teeth[i] <= lips[i] and teeth[i-1] > lips[i-1]) or \
                              (teeth[i] >= lips[i] and teeth[i-1] < lips[i-1])
            price_above_teeth = curr_close > teeth[i]
            
            if jaw_teeth_cross or teeth_lips_cross or price_above_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Alligator sleeping/awakening/eating conditions
            jaw_above_teeth = jaw[i] > teeth[i]
            teeth_above_lips = teeth[i] > lips[i]
            jaw_below_teeth = jaw[i] < teeth[i]
            teeth_below_lips = teeth[i] < lips[i]
            
            # Long when Jaw > Teeth > Lips (Alligator eating up) AND price > 1d EMA50 AND volume confirmation
            if jaw_above_teeth and teeth_above_lips and curr_close > curr_ema50_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Jaw < Teeth < Lips (Alligator eating down) AND price < 1d EMA50 AND volume confirmation
            elif jaw_below_teeth and teeth_below_lips and curr_close < curr_ema50_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals