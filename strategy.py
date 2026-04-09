#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v6
# Hypothesis: 6h strategy using weekly pivot points for trend direction and 6h Donchian channel breakouts with volume confirmation.
# Weekly pivot > price = bullish bias (long Donchian breakouts); Weekly pivot < price = bearish bias (short Donchian breakouts).
# Volume confirmation filters weak breakouts. Works in bull/bear via weekly pivot regime filter.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v6"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_point = (high_1w + low_1w + close_1w) / 3.0
    resistance_1 = 2 * pivot_point - low_1w
    support_1 = 2 * pivot_point - high_1w
    
    # Align HTF weekly pivot levels to 6h timeframe (wait for completed weekly bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    resistance_1_aligned = align_htf_to_ltf(prices, df_1w, resistance_1)
    support_1_aligned = align_htf_to_ltf(prices, df_1w, support_1)
    
    # 6h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(resistance_1_aligned[i]) or
            np.isnan(support_1_aligned[i]) or np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low
            if low[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high
            if high[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian high, volume confirmed, weekly pivot bias bullish
            if (high[i] > highest_high[i] and volume_confirmed and close[i] > pivot_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, volume confirmed, weekly pivot bias bearish
            elif (low[i] < lowest_low[i] and volume_confirmed and close[i] < pivot_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals