#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (Jaw/Teeth/Lips) with 12h EMA50 trend filter and volume confirmation.
# Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
# Long when Lips > Teeth > Jaw AND EMA50 rising AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when Lips < Teeth < Jaw AND EMA50 falling AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when Alligator alignment breaks or EMA50 flips direction or volume drops below average
# Designed for 6h timeframe with moderate trade frequency (target: 15-30/year) to avoid fee drag.
# Uses 12h EMA50 for trend filter to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
name = "6h_WilliamsAlligator_12hEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Alligator alignment
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    
    # EMA50 for trend filter (12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # EMA50 direction
    ema50_rising = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_12h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_12h_aligned[1:] > ema50_12h_aligned[:-1]
    ema50_falling[1:] = ema50_12h_aligned[1:] < ema50_12h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(lips_above_teeth[i]) or np.isnan(teeth_above_jaw[i]) or
            np.isnan(lips_below_teeth[i]) or np.isnan(teeth_below_jaw[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw AND EMA50 rising AND price > 12h EMA50 AND volume filter
            long_cond = (lips_above_teeth[i] and teeth_above_jaw[i] and 
                        ema50_rising[i] and (close[i] > ema50_12h_aligned[i]) and 
                        volume_filter[i])
            # Short conditions: Lips < Teeth < Jaw AND EMA50 falling AND price < 12h EMA50 AND volume filter
            short_cond = (lips_below_teeth[i] and teeth_below_jaw[i] and 
                         ema50_falling[i] and (close[i] < ema50_12h_aligned[i]) and 
                         volume_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks OR EMA50 falling OR volume filter fails
            if (not (lips_above_teeth[i] and teeth_above_jaw[i]) or 
                ema50_falling[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks OR EMA50 rising OR volume filter fails
            if (not (lips_below_teeth[i] and teeth_below_jaw[i]) or 
                ema50_rising[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals