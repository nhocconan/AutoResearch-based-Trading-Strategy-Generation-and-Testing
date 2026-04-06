#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h/1d filters for directional bias. 
# Uses 12h Donchian(20) breakout + 1d EMA(100) trend filter + volume confirmation (1.5x avg).
# Volume filter reduces false breakouts, trend filter avoids counter-trend trades.
# 12h provides structural context, 1d EMA filters trend, 6x executes.
# Target: 80-180 total trades over 4 years (20-45/year) to balance signal quality and fee drag.
# Works in bull via breakouts, in bear via trend filter preventing counter-trend entries.

name = "6h_donchian20_1dema100_vol_v1"
timeframe = "6h"
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
    
    # 12h Donchian channel (20-period) for breakout signals
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_12h = align_htf_to_ltf(prices, df_12h, high_20_12h)
    donchian_low_12h = align_htf_to_ltf(prices, df_12h, low_20_12h)
    donchian_mid_12h = (donchian_high_12h + donchian_low_12h) / 2
    
    # 1d EMA(100) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_100 = pd.Series(close_1d).ewm(span=100, adjust=False).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or 
            np.isnan(ema_100_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to 12h Donchian midpoint OR breaks below lower band
            if close[i] <= donchian_mid_12h[i] or close[i] < donchian_low_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to 12h Donchian midpoint OR breaks above upper band
            if close[i] >= donchian_mid_12h[i] or close[i] > donchian_high_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: 12h Donchian breakout + 1d EMA trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high_12h[i] and close[i] > ema_100_aligned[i]:
                    # Bullish breakout above 12h Donchian high with daily uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low_12h[i] and close[i] < ema_100_aligned[i]:
                    # Bearish breakdown below 12h Donchian low with daily downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals