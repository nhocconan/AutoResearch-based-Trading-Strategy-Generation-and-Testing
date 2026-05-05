#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation (2.0x)
# Long when price > Alligator Jaw (13-period SMMA smoothed 8) AND price > Alligator Teeth (8-period SMMA smoothed 5) AND price > Alligator Lips (5-period SMMA smoothed 3) AND price > 1d EMA34 AND volume > 2.0x 20-period average
# Short when price < Alligator Jaw AND price < Alligator Teeth AND price < Alligator Lips AND price < 1d EMA34 AND volume > 2.0x 20-period average
# Exit when price crosses Alligator Teeth OR 1d EMA34 filter reverses
# Uses Williams Alligator for trend identification + volume confirmation to reduce false signals
# 1d EMA34 provides higher timeframe trend filter effective in both bull and bear markets
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 12h (primary), HTF: 1w/1d

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_2.0x"
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
    
    # Get 12h data ONCE before loop for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    median_12h = (high_12h + low_12h) / 2.0  # Typical price for Alligator
    
    # Calculate Williams Alligator components (SMMA = Smoothed Moving Average)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator Jaw: 13-period SMMA of median price, shifted 8 bars
    jaw_raw = smma(median_12h, 13)
    jaw = np.roll(jaw_raw, 8)  # Shifted 8 bars into future
    
    # Alligator Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = smma(median_12h, 8)
    teeth = np.roll(teeth_raw, 5)  # Shifted 5 bars into future
    
    # Alligator Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = smma(median_12h, 5)
    lips = np.roll(lips_raw, 3)  # Shifted 3 bars into future
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation on 12h (threshold: 2.0x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Alligator Jaw, Teeth, Lips AND price > EMA34 AND volume spike
            if (close[i] > jaw_aligned[i] and 
                close[i] > teeth_aligned[i] and 
                close[i] > lips_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Alligator Jaw, Teeth, Lips AND price < EMA34 AND volume spike
            elif (close[i] < jaw_aligned[i] and 
                  close[i] < teeth_aligned[i] and 
                  close[i] < lips_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Alligator Teeth OR price < EMA34 (trend weakening)
            if close[i] < teeth_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Alligator Teeth OR price > EMA34 (trend weakening)
            if close[i] > teeth_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals