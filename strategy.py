#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel (trend filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian channels to daily timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    
    # Get daily data for price action
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period) for volatility filter
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to daily timeframe (1:1 mapping, but using proper function)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 50-day SMA for trend filter
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(sma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 50-day SMA
        uptrend = close_1d[i] > sma_50_aligned[i]
        downtrend = close_1d[i] < sma_50_aligned[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_14_aligned[i] > 0  # Always true if ATR calculated properly
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high + uptrend + volatility
            if close_1d[i] > donchian_high_20_aligned[i] and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian low + downtrend + volatility
            elif close_1d[i] < donchian_low_20_aligned[i] and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low
            if close_1d[i] < donchian_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high
            if close_1d[i] > donchian_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0