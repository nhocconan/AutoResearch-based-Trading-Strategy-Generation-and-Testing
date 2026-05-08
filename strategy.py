#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Long when Alligator jaws (13-period SMMA) crosses above teeth (8-period SMMA) AND price > EMA50(1d) AND volume > 1.5x 20-period average.
# Short when jaws crosses below teeth AND price < EMA50(1d) AND volume > 1.5x 20-period average.
# Exit when jaw crosses back over teeth (opposite crossover).
# Williams Alligator uses smoothed moving averages (SMMA) to filter noise and identify trends.
# Williams Alligator on 6h filters short-term noise; EMA50 on 1d filters long-term trend direction.
# Volume confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) as used in Williams Alligator."""
    sma = pd.Series(source).rolling(window=length, min_periods=length).mean()
    smma_val = np.full_like(source, np.nan, dtype=float)
    for i in range(length - 1, len(source)):
        if i == length - 1:
            smma_val[i] = sma.iloc[i]
        else:
            smma_val[i] = (smma_val[i - 1] * (length - 1) + source[i]) / length
    return smma_val

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6s volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 6h data for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Williams Alligator components on 6h
    median_price = (df_6h['high'].values + df_6h['low'].values) / 2
    jaw = smma(median_price, 13)  # Blue line (13-period SMMA)
    teeth = smma(median_price, 8)  # Red line (8-period SMMA)
    lips = smma(median_price, 5)   # Green line (5-period SMMA) - not used in crossover
    
    # Align 6h indicators to 6h timeframe (no alignment needed as already 6h)
    jaw_aligned = jaw
    teeth_aligned = teeth
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # EMA50 on 1d close
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50 and SMMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: jaw crosses above teeth, price > EMA50, volume filter
            jaw_above_teeth = jaw_aligned[i] > teeth_aligned[i]
            jaw_below_teeth_prev = jaw_aligned[i-1] <= teeth_aligned[i-1]
            long_crossover = jaw_above_teeth and jaw_below_teeth_prev
            
            long_cond = long_crossover and (close[i] > ema_50_aligned[i]) and volume_filter[i]
            # Short conditions: jaw crosses below teeth, price < EMA50, volume filter
            jaw_below_teeth = jaw_aligned[i] < teeth_aligned[i]
            jaw_above_teeth_prev = jaw_aligned[i-1] >= teeth_aligned[i-1]
            short_crossover = jaw_below_teeth and jaw_above_teeth_prev
            
            short_cond = short_crossover and (close[i] < ema_50_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: jaw crosses back below teeth
            jaw_below_teeth = jaw_aligned[i] < teeth_aligned[i]
            jaw_above_teeth_prev = jaw_aligned[i-1] >= teeth_aligned[i-1]
            exit_crossover = jaw_below_teeth and jaw_above_teeth_prev
            
            if exit_crossover:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: jaw crosses back above teeth
            jaw_above_teeth = jaw_aligned[i] > teeth_aligned[i]
            jaw_below_teeth_prev = jaw_aligned[i-1] <= teeth_aligned[i-1]
            exit_crossover = jaw_above_teeth and jaw_below_teeth_prev
            
            if exit_crossover:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals