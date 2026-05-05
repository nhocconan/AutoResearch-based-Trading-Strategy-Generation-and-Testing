#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume spike and 1d EMA50 trend filter
# Long when: price breaks above upper Donchian(20), volume > 2.0x 24-period average (12h equivalent), and close > 1d EMA50
# Short when: price breaks below lower Donchian(20), volume > 2.0x 24-period average, and close < 1d EMA50
# Exit when price returns to the opposite Donchian level (mean reversion)
# Uses Donchian channels from 6h for structure, effective in both bull (breakout continuation) and bear (mean reversion via exits) markets.
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian20_Breakout_1dEMA50_12hVolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate Donchian(20) on 6h
    if len(high) >= 20:
        upper_donchian = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_donchian = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        upper_donchian = np.full(n, np.nan)
        lower_donchian = np.full(n, np.nan)
    
    # Calculate volume confirmation on 6h using 24-period MA (equivalent to 12h lookback)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (2.0 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, volume filter, and above 1d EMA50
            if (close[i] > upper_donchian[i] and 
                open_price[i] <= upper_donchian[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian, volume filter, and below 1d EMA50
            elif (close[i] < lower_donchian[i] and 
                  open_price[i] >= lower_donchian[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below lower Donchian (mean reversion)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above upper Donchian (mean reversion)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals