#!/usr/bin/env python3
name = "6h_Donchian20_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h OHLC for Donchian and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian channel (20-period) on 12h
    upper_12h = np.full(len(high_12h), np.nan)
    lower_12h = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        upper_12h[i] = np.max(high_12h[i-20:i])
        lower_12h[i] = np.min(low_12h[i-20:i])
    
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 2.5x 24-period average (6h bars = 4 days)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (2.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # ~2 days (8*6h) to prevent overtrading
    
    start_idx = max(24, 20*2)  # Ensure Donchian and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above 12h Donchian upper with volume in 12h uptrend
            if (close[i] > upper_12h_aligned[i] and 
                close_12h[-1] > ema_50_12h_aligned[i] and  # Current 12h close above EMA50
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below 12h Donchian lower with volume in 12h downtrend
            elif (close[i] < lower_12h_aligned[i] and 
                  close_12h[-1] < ema_50_12h_aligned[i] and  # Current 12h close below EMA50
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below 12h Donchian lower or 12h trend changes to down
            if close[i] < lower_12h_aligned[i] or close_12h[-1] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above 12h Donchian upper or 12h trend changes to up
            if close[i] > upper_12h_aligned[i] or close_12h[-1] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 6h timeframe, price breaking above/below 12h Donchian(20) levels with volume confirmation and 12h EMA50 trend filter captures institutional breakout momentum. Works in bull markets (breakouts above upper band in 12h uptrend) and bear markets (breakdowns below lower band in 12h downtrend). Volume filter ensures participation, trend filter aligns with higher timeframe momentum. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing significant moves. Donchian channels provide objective breakout levels with institutional relevance.