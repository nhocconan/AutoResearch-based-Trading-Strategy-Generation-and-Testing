#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA20 trend filter and volume confirmation
# Long when price breaks above 20-day high with weekly EMA20 uptrend and volume > 1.5x average
# Short when price breaks below 20-day low with weekly EMA20 downtrend and volume > 1.5x average
# Exit when price retraces to 10-day SMA or reverses to opposite side of 10-day range
# Uses Donchian channels for breakout structure, EMA for trend, volume for conviction
# Designed to capture medium-term trends with low frequency to avoid fee drag
# Target: 30-80 total trades over 4 years (7-20/year) with size 0.25

name = "1d_Donchian20_WeeklyEMA20_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d 10-period SMA for exit
    sma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1w EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(sma10[i]) or np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, weekly EMA uptrend, volume confirmation
            if (close[i] > donchian_high[i] and 
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, weekly EMA downtrend, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retraces to 10-day SMA or reverses below Donchian mid
            if (close[i] <= sma10[i]) or (close[i] < donchian_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retraces to 10-day SMA or reverses above Donchian mid
            if (close[i] >= sma10[i]) or (close[i] > donchian_mid[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals