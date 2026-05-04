#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (JAWS/TEETH/LIPS) with 1d EMA34 trend filter and volume confirmation
# Uses Alligator crossover signals from completed 6h bars filtered by 1d EMA34 direction
# Volume confirmation (>1.8x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-150 total trades over 4 years = 19-37/year for 6h.
# 1d EMA34 ensures we only trade with the higher timeframe trend, reducing whipsaw in both bull and bear markets.

name = "6h_WilliamsAlligator_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 6h data for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    median_6h = (df_6h['high'].values + df_6h['low'].values) / 2.0
    
    # Calculate Williams Alligator lines
    jaws = pd.Series(median_6h).rolling(window=13, min_periods=13).mean().values
    jaws = pd.Series(jaws).rolling(window=8, min_periods=8).mean().values  # SMMA(13,8)
    
    teeth = pd.Series(median_6h).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth).rolling(window=5, min_periods=5).mean().values  # SMMA(8,5)
    
    lips = pd.Series(median_6h).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips).rolling(window=3, min_periods=3).mean().values  # SMMA(5,3)
    
    # Shift by 1 to use only completed 6h bar (avoid look-ahead)
    jaws_shifted = np.roll(jaws, 1)
    teeth_shifted = np.roll(teeth, 1)
    lips_shifted = np.roll(lips, 1)
    jaws_shifted[0] = np.nan
    teeth_shifted[0] = np.nan
    lips_shifted[0] = np.nan
    
    jaws_aligned = align_htf_to_ltf(prices, df_6h, jaws_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaws (bullish alignment) + price above 1d EMA34 + volume spike
            if (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaws (bearish alignment) + price below 1d EMA34 + volume spike
            elif (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaws_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines cross (Lips < Teeth) OR price crosses below 1d EMA34
            if lips_aligned[i] < teeth_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines cross (Lips > Teeth) OR price crosses above 1d EMA34
            if lips_aligned[i] > teeth_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals