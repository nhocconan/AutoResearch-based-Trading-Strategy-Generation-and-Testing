#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
# Long when Alligator jaws (13-period SMA shifted 8) crosses above teeth (8-period SMA shifted 5),
# 1w EMA50 rising, and volume > 1.5x 20-period average.
# Short when jaws cross below teeth, 1w EMA50 falling, and volume > 1.5x 20-period average.
# Exit when jaws cross back inside the teeth/lips range.
# Williams Alligator identifies trend phases. 1w EMA50 filters higher timeframe trend.
# Volume confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsAlligator_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator components (all calculated on 12h data)
    # Jaws: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward
    # Lips: 5-period SMMA shifted 3 bars forward
    def smma(source, period):
        """Smoothed Moving Average"""
        sma = pd.Series(source).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(source, np.nan, dtype=float)
        if len(source) >= period:
            smma_vals[period-1] = sma[period-1]
            for i in range(period, len(source)):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + source[i]) / period
        return smma_vals
    
    jaws_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply shifts (Williams Alligator specific)
    jaws = np.full_like(close, np.nan)
    teeth = np.full_like(close, np.nan)
    lips = np.full_like(close, np.nan)
    
    # Jaws shifted 8 bars forward
    if len(jaws_raw) > 8:
        jaws[8:] = jaws_raw[:-8]
    # Teeth shifted 5 bars forward
    if len(teeth_raw) > 5:
        teeth[5:] = teeth_raw[:-5]
    # Lips shifted 3 bars forward
    if len(lips_raw) > 3:
        lips[3:] = lips_raw[:-3]
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w EMA50 direction
    ema50_rising = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1w_aligned[1:] > ema50_1w_aligned[:-1]
    ema50_falling[1:] = ema50_1w_aligned[1:] < ema50_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Sufficient warmup for SMMA and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: jaws cross above teeth, 1w EMA50 rising, volume filter
            jaw_teeth_cross_up = (jaws[i] > teeth[i]) and (jaws[i-1] <= teeth[i-1]) if i > 0 else False
            long_cond = jaw_teeth_cross_up and ema50_rising[i] and volume_filter[i]
            # Short conditions: jaws cross below teeth, 1w EMA50 falling, volume filter
            jaw_teeth_cross_down = (jaws[i] < teeth[i]) and (jaws[i-1] >= teeth[i-1]) if i > 0 else False
            short_cond = jaw_teeth_cross_down and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: jaws cross back below teeth (trend weakening)
            jaw_teeth_cross_down = (jaws[i] < teeth[i]) and (jaws[i-1] >= teeth[i-1]) if i > 0 else False
            if jaw_teeth_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: jaws cross back above teeth (trend weakening)
            jaw_teeth_cross_up = (jaws[i] > teeth[i]) and (jaws[i-1] <= teeth[i-1]) if i > 0 else False
            if jaw_teeth_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals