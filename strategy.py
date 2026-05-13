#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 6h Donchian upper channel and close > 1w EMA50 with volume > 2.0x 20-bar average.
# Short when price breaks below 6h Donchian lower channel and close < 1w EMA50 with volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# 1w EMA50 provides strong higher timeframe trend alignment; volume spike confirms momentum.
# This strategy aims for fewer, higher-quality trades to avoid fee drag while maintaining edge in both bull and bear markets.

name = "6h_Donchian20_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_lower = low_series.rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper, close > 1w EMA50, volume spike
            if (high[i] > donchian_upper[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower, close < 1w EMA50, volume spike
            elif (low[i] < donchian_lower[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower OR volume drops below average
            if (low[i] < donchian_lower[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper OR volume drops below average
            if (high[i] > donchian_upper[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals