#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day volume confirmation and 1-week trend filter.
# Long when price breaks above Donchian(20) high, weekly trend bullish (price > weekly SMA50), volume > 1.5x 4h volume average.
# Short when price breaks below Donchian(20) low, weekly trend bearish (price < weekly SMA50), volume > 1.5x 4h volume average.
# Exit when price crosses back through Donchian midpoint.
# Uses weekly trend to filter for major trend direction, reducing whipsaw in ranging markets.
# Target: 20-30 trades/year per symbol.
name = "4h_Donchian20_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    # 1-week SMA50 for trend filter
    sma50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # 4h volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(sma50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donchian_high = high_roll[i]
        donchian_low = low_roll[i]
        weekly_sma50 = sma50_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: bullish breakout with weekly trend confirmation
            if (price > donchian_high and 
                price > weekly_sma50 and 
                vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish breakout with weekly trend confirmation
            elif (price < donchian_low and 
                  price < weekly_sma50 and 
                  vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint
            if price < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint
            if price > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals