#!/usr/bin/env python3
# 1d_1w_donchian_breakout_v2
# Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high, weekly close above weekly SMA50, and volume > 1.5x average.
# Short when price breaks below 20-day low, weekly close below weekly SMA50, and volume > 1.5x average.
# Exit when price returns to the 10-day moving average.
# Designed to generate ~10-20 trades/year per symbol to minimize fee decay while capturing strong trends.
# Works in both bull (breakouts) and bear (breakdowns) markets with volatility filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Daily 10-period SMA for exit
    sma_10 = np.full(n, np.nan)
    for i in range(9, n):
        sma_10[i] = np.mean(close[i-9:i+1])
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly 50-period SMA for trend filter
    sma_50_1w = np.full(len(close_1w), np.nan)
    for i in range(49, len(close_1w)):
        sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    # Align weekly SMA to daily timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Calculate average volume (50-period) for volume filter
    avg_volume = np.full(n, np.nan)
    for i in range(49, n):
        avg_volume[i] = np.mean(volume[i-49:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(sma_10[i]) or np.isnan(sma_50_1w_aligned[i]) or
            np.isnan(avg_volume[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 1:  # Long
            # Exit: price returns to 10-day SMA
            if price <= sma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 10-day SMA
            if price >= sma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions
            vol_filter = vol > 1.5 * avg_volume[i]
            
            # Bullish: price breaks above Donchian high, weekly trend up, volume confirmation
            if price > donchian_high[i] and close > sma_50_1w_aligned[i] and vol_filter:
                position = 1
                signals[i] = 0.25
            # Bearish: price breaks below Donchian low, weekly trend down, volume confirmation
            elif price < donchian_low[i] and close < sma_50_1w_aligned[i] and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals