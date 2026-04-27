#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Williams Alligator (Jaws=13, Teeth=8, Lips=5) with 1-week EMA34 trend filter and volume confirmation.
# The Alligator identifies trending vs ranging markets via smoothed moving averages.
# Long when Lips > Teeth > Jaws (bullish alignment) and price above Teeth, with weekly uptrend and volume confirmation.
# Short when Lips < Teeth < Jaws (bearish alignment) and price below Teeth, with weekly downtrend and volume confirmation.
# Designed for low trade frequency (target: 30-100 total trades over 4 years) to minimize fee drift.
# Works in bull markets (captures sustained uptrends) and bear markets (captures sustained downtrends) by only trading in strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma((high_1d + low_1d) / 2, 13)  # Jaw (blue line)
    teeth = smma((high_1d + low_1d) / 2, 8)   # Teeth (red line)
    lips = smma((high_1d + low_1d) / 2, 5)    # Lips (green line)
    
    # Align Alligator lines to 15m timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: Lips > Teeth > Jaws
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaws_aligned[i]
        # Bearish alignment: Lips < Teeth < Jaws
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaws_aligned[i]
        
        # Long condition: bullish alignment, price above Teeth, weekly uptrend, volume
        if (bullish_alignment and 
            close[i] > teeth_aligned[i] and 
            close[i] > ema34_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: bearish alignment, price below Teeth, weekly downtrend, volume
        elif (bearish_alignment and 
              close[i] < teeth_aligned[i] and 
              close[i] < ema34_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: alignment breakdown
        elif position == 1 and not bullish_alignment:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not bearish_alignment:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WilliamsAlligator_1wEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0