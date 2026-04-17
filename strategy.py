#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Daily 20-bar Donchian channel breakout with volume confirmation and 1-week trend filter
# Strategy logic: 
# - Long when price breaks above 20-day high with volume > 1.5x 10-day average and weekly close > weekly SMA50
# - Short when price breaks below 20-day low with volume > 1.5x 10-day average and weekly close < weekly SMA50
# - Exit when price returns to 10-day SMA or volatility filter fails
# - Designed for low trade frequency (<25/year) to minimize fee drag in bear markets
# - Uses 1d timeframe with 1w trend filter for multi-timeframe confluence

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA50 for trend filter
    if len(close_1w) >= 50:
        sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
        sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    else:
        sma50_1w_aligned = np.full(n, np.nan)
    
    # Calculate daily 20-bar Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-day SMA for exit
    sma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 1.5x 10-day average
    volume_ma10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(sma10[i]) or 
            np.isnan(volume_ma10[i]) or
            np.isnan(sma50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 10-day average
        volume_filter = volume[i] > (1.5 * volume_ma10[i])
        
        if position == 0:
            # Long: price breaks above 20-day high with volume and weekly uptrend
            if close[i] > high_20[i] and volume_filter and close[i] > sma50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume and weekly downtrend
            elif close[i] < low_20[i] and volume_filter and close[i] < sma50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 10-day SMA or weekly trend turns down
            if close[i] < sma10[i] or close[i] < sma50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 10-day SMA or weekly trend turns up
            if close[i] > sma10[i] or close[i] > sma50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0