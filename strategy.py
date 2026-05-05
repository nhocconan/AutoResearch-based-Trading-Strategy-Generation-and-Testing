#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Long when: Alligator jaws (13) < teeth (8) < lips (5) AND price > 1d EMA50 AND volume > 1.5x 20-period MA
# Short when: Alligator jaws (13) > teeth (8) > lips (5) AND price < 1d EMA50 AND volume > 1.5x 20-period MA
# Exit when: Alligator lines cross (jaws/teeth/lips no longer aligned) OR price crosses 1d EMA50
# Uses Williams Alligator for trend identification, 1d EMA50 for higher timeframe trend, volume for confirmation
# Timeframe: 4h, HTF: 1d. Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag.

name = "4h_Williams_Alligator_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams Alligator on 4h: SMAs with specific offsets
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    # Using SMA as approximation for SMMA (simple moving average)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or 
            np.isnan(lips_values[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator aligned bullish (jaws < teeth < lips) + above EMA50 + volume spike
            if (jaw_values[i] < teeth_values[i] < lips_values[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator aligned bearish (jaws > teeth > lips) + below EMA50 + volume spike
            elif (jaw_values[i] > teeth_values[i] > lips_values[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment broken OR price crosses below EMA50
            if not (jaw_values[i] < teeth_values[i] < lips_values[i]) or close[i] <= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment broken OR price crosses above EMA50
            if not (jaw_values[i] > teeth_values[i] > lips_values[i]) or close[i] >= ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals