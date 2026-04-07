#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Donchian breakout captures momentum moves, 1d EMA filter ensures alignment with daily trend
# Volume confirmation avoids false breakouts. Designed for low frequency: 12-37 trades/year
# to minimize fee drag in 12h timeframe, works in both bull and bear via trend filter.

name = "12h_donchian20_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period)
    high_max = np.zeros(n)
    low_min = np.zeros(n)
    for i in range(20, n):
        high_max[i] = np.max(high[i-19:i+1])
        low_min[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation (20-period average)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower band or trend fails
            if close[i] < low_min[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper band or trend fails
            if close[i] > high_max[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above Donchian upper band AND uptrend AND volume confirmation
            if close[i] > high_max[i] and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower band AND downtrend AND volume confirmation
            elif close[i] < low_min[i] and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals