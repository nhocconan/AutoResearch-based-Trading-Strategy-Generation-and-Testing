#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Volume Confirmation and Chop Filter
# Hypothesis: Donchian breakouts capture strong trends. Volume confirms institutional participation.
# Choppiness filter avoids whipsaws in sideways markets. Works in both bull and bear markets:
# In bull, breakouts above upper band continue up; breakdowns below lower band get bought (mean reversion).
# In bear, breakdowns below lower band continue down; breakouts above upper band get sold (mean reversion).
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Chop filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate Chop on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_12h
    
    # ATR(14) and sum
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chop value
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    # Align Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Donchian channels on 4h (20-period)
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(chop_aligned[i]) or np.isnan(high_4h[i]) or 
            np.isnan(low_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: avoid extreme chop (chop > 61.8 = ranging, chop < 38.2 = trending)
        # We want trending markets: chop < 61.8
        if chop_aligned[i] > 61.8:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below lower Donchian channel
            if close[i] < low_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above upper Donchian channel
            if close[i] > high_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above upper Donchian with volume
            if high[i] > high_4h[i] and close[i] > high_4h[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower Donchian with volume
            elif low[i] < low_4h[i] and close[i] < low_4h[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals