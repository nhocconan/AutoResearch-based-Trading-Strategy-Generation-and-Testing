#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Long when price breaks above Donchian(20) high AND weekly pivot bias is bullish (close > weekly pivot) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below Donchian(20) low AND weekly pivot bias is bearish (close < weekly pivot) AND volume > 2.0 * 20-bar avg volume
# Exit when price returns to Donchian(20) midpoint (mean reversion) or weekly pivot bias flips
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Donchian channels provide clear breakout levels; weekly pivot gives higher-timeframe bias; volume confirms institutional participation

name = "6h_Donchian20_WeeklyPivot_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Get weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly bias: bullish if close > pivot, bearish if close < pivot
    weekly_bias_bullish = weekly_close > weekly_pivot
    weekly_bias_bearish = weekly_close < weekly_pivot
    
    # Align weekly indicators to 6h timeframe (wait for completed weekly bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_bias_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bullish.astype(float))
    weekly_bias_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_bearish.astype(float))
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_bias_bullish_aligned[i]) or 
            np.isnan(weekly_bias_bearish_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Donchian breakout with weekly bias and volume confirmation
            # Long: break above Donchian high AND weekly bullish bias AND volume spike
            if close[i] > donchian_high[i] and weekly_bias_bullish_aligned[i] > 0.5 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND weekly bearish bias AND volume spike
            elif close[i] < donchian_low[i] and weekly_bias_bearish_aligned[i] > 0.5 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR weekly bias flips to bearish
            if close[i] <= donchian_mid[i] or weekly_bias_bullish_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR weekly bias flips to bullish
            if close[i] >= donchian_mid[i] or weekly_bias_bearish_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals