#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Uses 1d EMA34 for trend direction and Williams Alligator (JAW/TEETH/LIPS) from 6h for entry
# Volume confirmation requires 2.0x average volume to ensure strong participation
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 6h timeframe
# Works in both bull and bear markets by following the 1d trend direction and using Alligator for structure
# Prioritizes BTC/ETH performance with SOL as secondary

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator from 6h data (Smoothed Medians)
    # JAW: 13-period SMMA, shifted 8 bars ahead
    # TEETH: 8-period SMMA, shifted 5 bars ahead  
    # LIPS: 5-period SMMA, shifted 3 bars ahead
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition (JAW 8, TEETH 5, LIPS 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Williams Alligator signals with 1d trend filter
        # Alligator sleeping: jaws, teeth, lips intertwined (no clear trend)
        # Alligator awakening: lines diverge in direction of trend
        # Alligator eating: lines diverge widely, strong trend
        
        # Bullish: Lips > Teeth > Jaw (green alignment) + price above all + 1d uptrend + volume spike
        # Bearish: Lips < Teeth < Jaw (red alignment) + price below all + 1d downtrend + volume spike
        
        if position == 0:
            # Bullish entry: Lips > Teeth > Jaw AND price > Lips AND 1d uptrend AND volume spike
            if (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and 
                close[i] > lips_shifted[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Bearish entry: Lips < Teeth < Jaw AND price < Lips AND 1d downtrend AND volume spike
            elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and 
                  close[i] < lips_shifted[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines cross back (Lips < Teeth) OR price below 1d EMA34 (trend change)
            if lips_shifted[i] < teeth_shifted[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines cross back (Lips > Teeth) OR price above 1d EMA34 (trend change)
            if lips_shifted[i] > teeth_shifted[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals