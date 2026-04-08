#!/usr/bin/env python3
# 1d_1w_donchian20_volume_sma_filter_v1
# Hypothesis: 1-day Donchian(20) breakouts with volume confirmation and SMA(50) trend filter.
# Long when price breaks above 20-day high with volume > 1.5x average and price > SMA50.
# Short when price breaks below 20-day low with volume > 1.5x average and price < SMA50.
# Exit when price crosses the opposite Donchian level (20-day low for longs, 20-day high for shorts).
# Uses weekly timeframe for trend context: only take longs when weekly close > weekly SMA50, shorts when weekly close < weekly SMA50.
# Target: 10-25 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian20_volume_sma_filter_v1"
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
    
    # Get weekly data for trend filter (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend filter
    weekly_close = df_1w['close'].values
    weekly_sma50 = np.full(len(weekly_close), np.nan)
    for i in range(50-1, len(weekly_close)):
        weekly_sma50[i] = np.mean(weekly_close[i-50+1:i+1])
    
    # Align weekly SMA50 to daily timeframe (wait for weekly bar to close)
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    
    # Calculate 20-day Donchian channels (highest high, lowest low over 20 days)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20-1, n):
        donchian_high[i] = np.max(high[i-20+1:i+1])
        donchian_low[i] = np.min(low[i-20+1:i+1])
    
    # Calculate 20-period volume average for confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_sma50_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below 20-day low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above 20-day high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above 20-day high, volume surge, price > SMA50, weekly close > weekly SMA50
            if (close[i] > donchian_high[i] and 
                vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i] and
                close[i] > weekly_sma50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below 20-day low, volume surge, price < SMA50, weekly close < weekly SMA50
            elif (close[i] < donchian_low[i] and 
                  vol_ma[i] > 0 and volume[i] > 1.5 * vol_ma[i] and
                  close[i] < weekly_sma50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals