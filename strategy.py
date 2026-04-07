#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation
# Donchian channels provide clear breakout signals in both trending and ranging markets
# Daily EMA200 filter ensures alignment with higher timeframe trend
# Volume surge confirms institutional participation
# Low frequency design: targets 20-40 trades per year to minimize fee drag

name = "4h_donchian20_1d_trend_volume_v2"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema200_1d = close_1d.ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below daily EMA200
        uptrend = close[i] > ema200_1d_aligned[i]
        downtrend = close[i] < ema200_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower band or trend fails
            if close[i] < donchian_low[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper band or trend fails
            if close[i] > donchian_high[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above Donchian upper band AND uptrend AND volume confirmation
            if close[i] > donchian_high[i] and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower band AND downtrend AND volume confirmation
            elif close[i] < donchian_low[i] and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals