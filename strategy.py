#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Targets 12-37 trades per year (50-150 total over 4 years) to minimize fee drag
# Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence via SMAs
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume confirmation (2.0x 20-period average) filters low-momentum breakouts
# Uses discrete position sizing 0.25 to balance exposure and risk
# Works in both bull and bear: trend filter prevents counter-trend entries

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 6h (SMAs: Jaw=13*8, Teeth=8*8, Lips=5*8)
    # Alligator uses SMAs of median price (H+L)/2 with specific periods
    median_price = (high + low) / 2
    jaw_period = 13 * 8  # 104 periods
    teeth_period = 8 * 8  # 64 periods
    lips_period = 5 * 8   # 40 periods
    
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Calculate 6h volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and volume)
    start_idx = max(jaw_period, teeth_period, lips_period) + 20
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator sleeping: jaws, teeth, lips intertwined (no trend)
            # Alligator awakening: lips cross above/below teeth with divergence
            # Long: lips cross above teeth AND above jaw AND price > 1d EMA50 AND volume confirm
            if (lips[i] > teeth[i] and lips[i-1] <= teeth[i-1] and  # lips crossing up teeth
                lips[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: lips cross below teeth AND below jaw AND price < 1d EMA50 AND volume confirm
            elif (lips[i] < teeth[i] and lips[i-1] >= teeth[i-1] and  # lips crossing down teeth
                  lips[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: lips cross below teeth OR price < 1d EMA50
            if (lips[i] < teeth[i] and lips[i-1] >= teeth[i-1] or  # lips crossing down teeth
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: lips cross above teeth OR price > 1d EMA50
            if (lips[i] > teeth[i] and lips[i-1] <= teeth[i-1] or  # lips crossing up teeth
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals