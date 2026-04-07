#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-day trend filter and volume confirmation
# Donchian breakouts capture momentum in trending markets. Daily trend filter ensures
# alignment with higher timeframe direction. Volume confirmation filters low-participation moves.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Designed for low frequency in 12h timeframe to minimize fee drag.

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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema20_1d = close_1d.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate Donchian Channel (20-period) on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_max.values
    donchian_low = low_min.values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend: close above/below daily EMA20
        daily_uptrend = close[i] > ema20_1d_aligned[i]
        daily_downtrend = close[i] < ema20_1d_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit if daily trend turns down or price breaks below Donchian low
            if not daily_uptrend or close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit if daily trend turns up or price breaks above Donchian high
            if not daily_downtrend or close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: daily uptrend + price breaks above Donchian high + volume confirmation
            if daily_uptrend and close[i] > donchian_high[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: daily downtrend + price breaks below Donchian low + volume confirmation
            elif daily_downtrend and close[i] < donchian_low[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals