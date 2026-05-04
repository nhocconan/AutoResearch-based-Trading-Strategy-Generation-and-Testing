#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends
# When Lips > Teeth > Jaw = uptrend (green), Lips < Teeth < Jaw = downtrend (red)
# Strong trend + volume confirmation = entry, opposite Alligator alignment = exit
# Works in both bull and bear markets by following the Alligator's trend direction
# Discrete sizing 0.25 targets 12-37 trades/year for 12h timeframe

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Williams Alligator components (using 12h prices)
    close_s = pd.Series(close)
    # Jaw: 13-period SMMA (smoothed moving average) - approx with EMA for simplicity
    jaw = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    # Teeth: 8-period SMMA
    teeth = close_s.ewm(span=8, adjust=False, min_periods=8).mean().values
    # Lips: 5-period SMMA
    lips = close_s.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (uptrend) AND 1d EMA34 uptrend AND volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (downtrend) AND 1d EMA34 downtrend AND volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR price closes below Jaw
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw) OR price closes above Jaw
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i] or close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals