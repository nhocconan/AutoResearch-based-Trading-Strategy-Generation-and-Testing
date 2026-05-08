#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mdt_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA(21) trend filter and volume confirmation.
# Long when price breaks above Donchian upper band (20-period) AND weekly EMA(21) above weekly close AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band (20-period) AND weekly EMA(21) below weekly close AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Donchian channel (between upper and lower bands).
# Donchian provides trend-following structure, weekly EMA filters higher timeframe bias, volume confirms institutional participation.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian_20_WeeklyEMA21_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) on close
    weekly_close = df_w['close'].values
    ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMA(21) to daily timeframe
    ema21_aligned = align_htf_to_ltf(prices, df_w, ema21)
    
    # Donchian channels (20-period) on daily data
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema21_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, weekly EMA above weekly close, volume filter
            long_cond = (close[i] > donchian_upper[i]) and (ema21_aligned[i] > weekly_close[i-1] if i > 0 else False) and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, weekly EMA below weekly close, volume filter
            short_cond = (close[i] < donchian_lower[i]) and (ema21_aligned[i] < weekly_close[i-1] if i > 0 else False) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower band
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper band
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals