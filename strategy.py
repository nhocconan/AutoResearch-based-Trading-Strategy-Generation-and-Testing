#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# This strategy trades when the Alligator lines are aligned (bullish/bearish) with
# the higher timeframe trend and volume confirmation. The Alligator (Jaw/Teeth/Lips)
# identifies trend direction and strength. Works in both bull and bear markets by
# following the 1d EMA50 trend direction. Uses discrete position sizing (0.25).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (13-period SMMA, 8-period shift)
    # Teeth (8-period SMMA, 5-period shift)
    # Lips (5-period SMMA, 3-period shift)
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    # We'll use EMA as approximation for SMMA
    
    # Calculate SMMA-like values using EMA with appropriate periods
    # Jaw: 13-period EMA shifted 8 bars
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean().values
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan
    
    # Teeth: 8-period EMA shifted 5 bars
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean().values
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan
    
    # Lips: 5-period EMA shifted 3 bars
    lips = pd.Series(close).ewm(span=5, adjust=False).mean().values
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator: Lips > Teeth > Jaw + above 1d EMA + volume spike
            if (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Bearish Alligator: Lips < Teeth < Jaw + below 1d EMA + volume spike
            elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross in opposite direction
            if position == 1:
                # Exit long: Lips cross below Jaw (trend weakening)
                if lips_shifted[i] < jaw_shifted[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Lips cross above Jaw (trend weakening)
                if lips_shifted[i] > jaw_shifted[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0