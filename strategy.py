#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian(20) high AND price > weekly pivot support AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND price < weekly pivot resistance AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Donchian channel (opposite band)
# Weekly pivot levels derived from previous week's high/low/close
# This captures strong trending moves with weekly structure context while avoiding counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels on 6h (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot points from previous week's OHLC
    # Pivot = (H + L + C) / 3
    # Support 1 = (2 * Pivot) - High
    # Resistance 1 = (2 * Pivot) - Low
    if len(df_1w) > 0:
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        pivot = (weekly_high + weekly_low + weekly_close) / 3
        support_1 = (2 * pivot) - weekly_high
        resistance_1 = (2 * pivot) - weekly_low
        
        # Align weekly pivot levels to 6h timeframe (with 1-bar delay for completed weekly bar)
        pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
        support_1_aligned = align_htf_to_ltf(prices, df_1w, support_1)
        resistance_1_aligned = align_htf_to_ltf(prices, df_1w, resistance_1)
    else:
        # Fallback if no weekly data
        pivot_aligned = np.full(n, np.nan)
        support_1_aligned = np.full(n, np.nan)
        resistance_1_aligned = np.full(n, np.nan)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Donchian + buffer)
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(support_1_aligned[i]) or 
            np.isnan(resistance_1_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above Donchian high AND above weekly support AND volume confirmation
            if (price > high_20[i] and price > support_1_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below Donchian low AND below weekly resistance AND volume confirmation
            elif (price < low_20[i] and price < resistance_1_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below Donchian low (opposite band)
            if price < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Donchian high (opposite band)
            if price > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0