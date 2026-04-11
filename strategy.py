#!/usr/bin/env python3
"""
6h_1w_1d_price_channel_v1
Strategy: 6h price channel breakout with weekly trend filter and daily volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Combines 6h Donchian(20) breakout with weekly trend direction (price above/below weekly SMA50) and daily volume surge (>1.5x average) to capture strong trend continuations. Works in bull markets via long breakouts above weekly SMA50 and in bear markets via short breakouts below weekly SMA50. Weekly filter ensures we only trade with the higher timeframe trend, reducing whipsaw. Volume confirmation ensures breakouts have conviction. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_price_channel_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly SMA50 for trend filter
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(sma_50_1w_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        vol_current = volume[i]
        
        # Trend filters
        above_weekly_sma = price_close > sma_50_1w_aligned[i]
        below_weekly_sma = price_close < sma_50_1w_aligned[i]
        
        # Volume confirmation
        volume_surge = vol_current > (1.5 * vol_avg_20_aligned[i])
        
        # Donchian breakout conditions
        breakout_long = price_close > highest_high[i]
        breakout_short = price_close < lowest_low[i]
        
        # Long: Donchian breakout above weekly SMA with volume surge
        long_signal = breakout_long and above_weekly_sma and volume_surge
        
        # Short: Donchian breakdown below weekly SMA with volume surge
        short_signal = breakout_short and below_weekly_sma and volume_surge
        
        # Exit when price returns to midpoint of channel (mean reversion)
        channel_mid = (highest_high[i] + lowest_low[i]) / 2.0
        exit_long = position == 1 and price_close < channel_mid
        exit_short = position == -1 and price_close > channel_mid
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Combines 6h Donchian(20) breakout with weekly trend direction (price above/below weekly SMA50) and daily volume surge (>1.5x average) to capture strong trend continuations. Works in bull markets via long breakouts above weekly SMA50 and in bear markets via short breakouts below weekly SMA50. Weekly filter ensures we only trade with the higher timeframe trend, reducing whipsaw. Volume confirmation ensures breakouts have conviction. Target: 50-150 total trades over 4 years.