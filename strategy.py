#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Targets 12-37 trades per year (50-150 total over 4 years) to minimize fee drag
# Williams Alligator (Jaw/Teeth/Lips) identifies trend presence and direction
# 1d EMA50 ensures alignment with daily higher timeframe trend
# Volume confirmation (1.5x 20-period average) filters false breakouts
# Uses discrete position sizing 0.25 to balance exposure and risk
# Works in both bull and bear: Alligator identifies trend, EMA filter prevents counter-trend

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe (using median prices)
    median_price = (high + low) / 2.0
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Smoothed median price (SMMA-like using EMA as approximation)
    smoothed_median = pd.Series(median_price).ewm(span=2, adjust=False).mean().values
    
    # Alligator lines: Jaw (blue), Teeth (red), Lips (green)
    jaw = pd.Series(smoothed_median).ewm(span=jaw_period, adjust=False).mean().shift(jaw_shift).values
    teeth = pd.Series(smoothed_median).ewm(span=teeth_period, adjust=False).mean().shift(teeth_shift).values
    lips = pd.Series(smoothed_median).ewm(span=lips_period, adjust=False).mean().shift(lips_shift).values
    
    # Calculate 12h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and volume)
    start_idx = max(jaw_period, teeth_period, lips_period) + jaw_shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: Lips > Teeth > Jaw = uptrend
            # Lips < Teeth < Jaw = downtrend
            # Long: Alligator bullish alignment AND price > 1d EMA50 AND volume confirm
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment AND price < 1d EMA50 AND volume confirm
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR price < 1d EMA50
            if (lips[i] < teeth[i] or teeth[i] < jaw[i] or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR price > 1d EMA50
            if (lips[i] > teeth[i] or teeth[i] > jaw[i] or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals