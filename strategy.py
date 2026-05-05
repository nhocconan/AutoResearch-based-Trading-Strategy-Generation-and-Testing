#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND 1d EMA34 uptrend AND volume > 1.5x 20-period average
# Short when Alligator jaws < teeth < lips AND 1d EMA34 downtrend AND volume > 1.5x 20-period average
# Exit when Alligator lines cross (jaws < teeth for longs, jaws > teeth for shorts) OR 1d trend reverses
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-35 trades/year per symbol.
# Williams Alligator identifies trending vs ranging markets via SMMA alignment.
# 1d EMA34 filter ensures higher timeframe alignment to avoid counter-trend whipsaws.
# Volume confirmation ensures institutional participation. Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "4h_WilliamsAlligator_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def smma(source, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 4h data
    median_price_4h = (df_4h['high'].values + df_4h['low'].values) / 2.0
    
    # Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMMA
    jaw = smma(median_price_4h, 13)  # Blue line
    teeth = smma(median_price_4h, 8)  # Red line
    lips = smma(median_price_4h, 5)   # Green line
    
    # Align Alligator lines to prices timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Uptrend when close > EMA34, downtrend when close < EMA34
    uptrend_1d = close_1d > ema_34_1d
    downtrend_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 4h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)  # No volume confirmation if insufficient data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator aligned (jaw > teeth > lips) AND 1d EMA34 uptrend AND volume spike
            if (jaw_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Alligator aligned (jaw < teeth < lips) AND 1d EMA34 downtrend AND volume spike
            elif (jaw_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < lips_aligned[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses alignment (jaw < teeth) OR 1d trend changes to downtrend
            if (jaw_aligned[i] < teeth_aligned[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses alignment (jaw > teeth) OR 1d trend changes to uptrend
            if (jaw_aligned[i] > teeth_aligned[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals