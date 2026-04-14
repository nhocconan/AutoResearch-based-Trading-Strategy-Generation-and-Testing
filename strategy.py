#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with daily volume confirmation and weekly trend filter
# Long when price breaks above 20-period Donchian high AND daily volume > 1.5x 20-day average AND price > weekly EMA20
# Short when price breaks below 20-period Donchian low AND daily volume > 1.5x 20-day average AND price < weekly EMA20
# Exit when price crosses back inside the Donchian channel
# Uses Donchian channels for breakout signals, volume for confirmation, weekly EMA for trend filter
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for volume confirmation and weekly EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period Donchian channels on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Donchian + buffer)
    start = 25
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: break above Donchian high + volume confirmation + above weekly EMA20
            if (price > donchian_high[i] and vol > vol_threshold and price > ema20_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low + volume confirmation + below weekly EMA20
            elif (price < donchian_low[i] and vol > vol_threshold and price < ema20_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back inside Donchian channel (below Donchian low)
            if price < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back inside Donchian channel (above Donchian high)
            if price > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_DailyVol_WeeklyEMA"
timeframe = "12h"
leverage = 1.0