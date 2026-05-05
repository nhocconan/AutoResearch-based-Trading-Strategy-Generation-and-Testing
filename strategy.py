#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h trend filter + volume confirmation
# Long when: price breaks above Donchian upper (20-period high) AND 12h EMA50 > EMA200 (uptrend) AND volume > 1.5x 24-period MA
# Short when: price breaks below Donchian lower (20-period low) AND 12h EMA50 < EMA200 (downtrend) AND volume > 1.5x 24-period MA
# Exit when: price crosses Donchian middle (10-period average of upper/lower) OR trend filter reverses
# Uses Donchian for breakouts, 12h EMA crossover for trend filter, volume for conviction
# Timeframe: 6h, HTF: 12h. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Donchian20_12hTrend_VolumeConfirm"
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
    
    # Calculate volume confirmation on 6h using 24-period MA (equivalent to 1d lookback)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (1.5 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian channels on 6h
    if len(high) >= 20 and len(low) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Get 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 and EMA200 for trend filter
    if len(close_12h) >= 200:
        ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
        trend_filter = ema_50_12h > ema_200_12h  # True for uptrend, False for downtrend
    else:
        ema_50_12h = np.full(len(close_12h), np.nan)
        ema_200_12h = np.full(len(close_12h), np.nan)
        trend_filter = np.full(len(close_12h), np.nan)
    
    # Align 12h trend filter to 6h timeframe
    trend_filter_aligned = align_htf_to_ltf(prices, df_12h, trend_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(trend_filter_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper + uptrend + volume filter
            if (close[i] > donchian_upper[i] and 
                trend_filter_aligned[i] and  # Uptrend
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower + downtrend + volume filter
            elif (close[i] < donchian_lower[i] and 
                  not trend_filter_aligned[i] and  # Downtrend
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses middle OR trend turns down
            if (close[i] < donchian_middle[i] or not trend_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses middle OR trend turns up
            if (close[i] > donchian_middle[i] or trend_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals