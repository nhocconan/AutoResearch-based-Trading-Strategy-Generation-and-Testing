#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-week trend filter and volume confirmation.
Long when price breaks above 20-period Donchian upper band + weekly close > weekly EMA50 + volume > 1.5x daily average.
Short when price breaks below 20-period Donchian lower band + weekly close < weekly EMA50 + volume > 1.5x daily average.
Exit when price crosses back through the Donchian middle band or weekly trend changes.
Targets 15-30 trades/year per symbol to avoid fee drag.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    wk_close = df_1w['close'].values
    wk_ema50 = pd.Series(wk_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    wk_ema50_aligned = align_htf_to_ltf(prices, df_1w, wk_ema50)
    wk_close_aligned = align_htf_to_ltf(prices, df_1w, wk_close)
    
    # Get daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i-donchian_len+1:i+1])
        lower[i] = np.min(low[i-donchian_len+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + weekly EMA (50) + volume MA (20)
    start_idx = max(donchian_len - 1, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(wk_ema50_aligned[i]) or np.isnan(wk_close_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Current levels
        upper_band = upper[i]
        lower_band = lower[i]
        middle_band = middle[i]
        weekly_ema = wk_ema50_aligned[i]
        weekly_close = wk_close_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above upper band + weekly uptrend + volume
            if price_now > upper_band and weekly_close > weekly_ema and vol_filter:
                signals[i] = size
                position = 1
            # Short: break below lower band + weekly downtrend + volume
            elif price_now < lower_band and weekly_close < weekly_ema and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below middle band OR weekly trend turns down
            if price_now < middle_band or weekly_close < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above middle band OR weekly trend turns up
            if price_now > middle_band or weekly_close > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_WeeklyEMA50_Volume"
timeframe = "12h"
leverage = 1.0