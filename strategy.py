#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trending vs ranging markets
# Enters on Alligator alignment in direction of 1d EMA34 trend with volume confirmation (>1.3x 20-period EMA volume)
# Exits when Alligator lines intertwine (market returns to ranging) or trend fails
# Discrete sizing 0.25 targets 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams Alligator excels in catching strong trends while avoiding choppy markets
# 1d EMA34 filter ensures alignment with higher timeframe trend, reducing counter-trend trades
# Works in both bull (Alligator eating with uptrend) and bear (Alligator sleeping then eating with downtrend) markets

name = "4h_WilliamsAlligator_1dEMA34_VolumeConfirm"
timeframe = "4h"
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
    if len(df_1d) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Williams Alligator components on 4h data (using prior completed bar)
    # Jaw: Blue line - 13-period SMMA shifted 8 bars
    # Teeth: Red line - 8-period SMMA shifted 5 bars  
    # Lips: Green line - 5-period SMMA shifted 3 bars
    median_4h = (high + low) / 2
    
    # Calculate SMMA (Smoothed Moving Average) - similar to Wilder's smoothing
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current price) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_4h, 13)
    teeth = smma(median_4h, 8)
    lips = smma(median_4h, 5)
    
    # Shift to use prior completed 4h bar (avoid look-ahead)
    jaw_shifted = np.roll(jaw, 1)
    teeth_shifted = np.roll(teeth, 1)
    lips_shifted = np.roll(lips, 1)
    jaw_shifted[0] = np.nan
    teeth_shifted[0] = np.nan
    lips_shifted[0] = np.nan
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator conditions:
        # Alligator sleeping (all lines intertwined) = no trade
        # Alligator eating (lines separated in order) = trend present
        # Long: Lips > Teeth > Jaw (green above red above blue) AND price > 1d EMA34 AND volume spike
        # Short: Lips < Teeth < Jaw (green below red below blue) AND price < 1d EMA34 AND volume spike
        
        if position == 0:
            # Check for Alligator eating (lines separated)
            lips_above_teeth = lips_shifted[i] > teeth_shifted[i]
            teeth_above_jaw = teeth_shifted[i] > jaw_shifted[i]
            lips_below_teeth = lips_shifted[i] < teeth_shifted[i]
            teeth_below_jaw = teeth_shifted[i] < jaw_shifted[i]
            
            # Long conditions: Alligator eating upwards AND price > 1d EMA34 AND volume spike
            if (lips_above_teeth and teeth_above_jaw and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > (1.3 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator eating downwards AND price < 1d EMA34 AND volume spike
            elif (lips_below_teeth and teeth_below_jaw and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > (1.3 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator starts sleeping (lines intertwine) OR trend fails
            lips_above_teeth = lips_shifted[i] > teeth_shifted[i]
            teeth_above_jaw = teeth_shifted[i] > jaw_shifted[i]
            # Exit if Alligator not eating upwards OR price crosses below 1d EMA34
            if not (lips_above_teeth and teeth_above_jaw) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator starts sleeping (lines intertwine) OR trend fails
            lips_below_teeth = lips_shifted[i] < teeth_shifted[i]
            teeth_below_jaw = teeth_shifted[i] < jaw_shifted[i]
            # Exit if Alligator not eating downwards OR price crosses above 1d EMA34
            if not (lips_below_teeth and teeth_below_jaw) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals