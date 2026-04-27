#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Williams Alligator (Jaw, Teeth, Lips) from 12h data: long when Lips > Teeth > Jaw with 1d uptrend and volume spike,
# short when Lips < Teeth < Jaw with 1d downtrend and volume spike.
# Williams Alligator helps identify trends and avoid whipsaws in ranging markets.
# Volume filter ensures trades occur during periods of high participation.
# Designed for 12-30 trades/year per symbol (48-120 total over 4 years) to minimize fee drag.
# Williams Alligator is effective in both bull and bear markets by filtering out chop.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 8:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
    # SMMA (Smoothed Moving Average) is similar to EMA but with different smoothing
    def smma(series, period):
        if len(series) < period:
            return np.full_like(series, np.nan)
        result = np.full_like(series, np.nan)
        # First value is SMA
        result[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Price) / Period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw_12h = smma(close_12h, 13)  # Blue line
    teeth_12h = smma(close_12h, 8)  # Red line
    lips_12h = smma(close_12h, 5)   # Green line
    
    # Shift the lines: Jaw by 8, Teeth by 5, Lips by 3
    jaw_12h = np.roll(jaw_12h, 8)
    teeth_12h = np.roll(teeth_12h, 5)
    lips_12h = np.roll(lips_12h, 3)
    
    # Align Williams Alligator lines to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND 1d uptrend AND volume spike
        if (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND 1d downtrend AND volume spike
        elif (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0