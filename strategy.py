#!/usr/bin/env python3
name = "1d_Donchian20_200EMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 210:  # Need enough data for 200 EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    trend_up = close > ema50_1w_aligned
    trend_down = close < ema50_1w_aligned
    
    # Daily Donchian channel (20-period)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    
    for i in range(n):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Daily 200 EMA trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema200[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma20[i]) or
            volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                pass  # Keep flat
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND weekly uptrend AND price > 200 EMA AND volume above average
            if (close[i] > donchian_high[i] and 
                trend_up[i] and 
                close[i] > ema200[i] and 
                volume[i] > vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND weekly downtrend AND price < 200 EMA AND volume above average
            elif (close[i] < donchian_low[i] and 
                  trend_down[i] and 
                  close[i] < ema200[i] and 
                  volume[i] > vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low OR weekly trend turns down
            if close[i] < donchian_low[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high OR weekly trend turns up
            if close[i] > donchian_high[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter, daily EMA200 filter, and volume confirmation.
# Enters long when price breaks above 20-day high in weekly uptrend above EMA200 with above-average volume.
# Enters short when price breaks below 20-day low in weekly downtrend below EMA200 with above-average volume.
# Exits when price breaks opposite Donchian band or weekly trend reverses.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# EMA200 filter prevents counter-trend trades against strong long-term trend.
# Volume confirmation ensures breakouts are supported by participation.
# Position size 0.25 limits risk. Designed for fewer trades (~10-25/year) to minimize fee drag.
# Works in bull markets (captures uptrend continuations) and bear markets (captures downtrends).