#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w EMA34 trend filter
# Long when: price breaks above upper Donchian channel (20-period high), volume > 1.8x 20-period average, and close > 1w EMA34
# Short when: price breaks below lower Donchian channel (20-period low), volume > 1.8x 20-period average, and close < 1w EMA34
# Exit when price returns to the opposite Donchian level (mean reversion) or opposite breakout
# Uses Donchian levels from 12h for structure, effective in both bull (breakout continuation) and bear (mean reversion via exits) markets.
# Timeframe: 12h, HTF: 1d/1w. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Donchian20_Breakout_1wEMA34_VolumeSpike"
timeframe = "12h"
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
    
    # Calculate volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels from previous 12h bar
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    # Using previous bar's values (shifted by 1) to avoid look-ahead
    if len(high) >= 21:
        # Rolling window of 20 on previous bars
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
        donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, volume filter, and above 1w EMA34
            if (close[i] > donchian_upper[i] and 
                open_price[i] <= donchian_upper[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian, volume filter, and below 1w EMA34
            elif (close[i] < donchian_lower[i] and 
                  open_price[i] >= donchian_lower[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below lower Donchian (mean reversion) or breaks below upper Donchian (failure)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above upper Donchian (mean reversion) or breaks above lower Donchian (failure)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals