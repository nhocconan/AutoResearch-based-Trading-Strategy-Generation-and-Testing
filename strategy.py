#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# - Weekly pivot levels computed from prior week's OHLC
# - Long when price breaks above Donchian upper band AND close > weekly pivot point AND volume > 1.5x average
# - Short when price breaks below Donchian lower band AND close < weekly pivot point AND volume > 1.5x average
# - Exit when price crosses weekly pivot point in opposite direction OR volume drops below 0.8x average
# - Weekly pivot filter ensures trades align with higher timeframe structure, reducing counter-trend whipsaws
# - Volume confirmation filters low-momentum breakouts
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - Works in bull markets via breakouts, in bear via short breakdowns with weekly pivot as trend filter

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Pre-compute weekly pivot points: (H+L+C)/3 from prior completed week
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot point for each week
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (completed weekly pivot only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Pre-compute Donchian channels (20-period) on 6h data
    lookback = 20
    highest_high = prices['high'].rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = prices['low'].rolling(window=lookback, min_periods=lookback).min().values
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    # Pre-compute volume filter: < 0.8x average volume for exit (loss of momentum)
    vol_weak = prices['volume'] < (0.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long breakout: price > Donchian upper band AND close > weekly pivot AND volume spike
            if (prices['close'].iloc[i] > highest_high[i] and 
                prices['close'].iloc[i] > weekly_pivot_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown: price < Donchian lower band AND close < weekly pivot AND volume spike
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  prices['close'].iloc[i] < weekly_pivot_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price crosses weekly pivot point in opposite direction
            # 2. Volume drops below 0.8x average (loss of momentum)
            if position == 1:  # Long position
                if (prices['close'].iloc[i] < weekly_pivot_aligned[i] or 
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (prices['close'].iloc[i] > weekly_pivot_aligned[i] or 
                    vol_weak.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals