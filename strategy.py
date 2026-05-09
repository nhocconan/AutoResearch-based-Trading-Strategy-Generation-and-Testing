#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 1d volume confirmation and 1d trend filter.
# Long when price breaks above Donchian(20) high with volume > 1.5x 20-period EMA and close > 1d EMA50.
# Short when price breaks below Donchian(20) low with volume > 1.5x 20-period EMA and close < 1d EMA50.
# Exit when price crosses back through the Donchian midpoint.
# Designed to capture strong trends in both bull and bear markets with volume confirmation to avoid false breakouts.
# Uses 1d EMA50 for trend filter to ensure alignment with daily trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.

name = "6h_Donchian20_1dVolume_EMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 6h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Donchian midpoint for exit
    midpoint = (highest_high + lowest_low) / 2.0
    
    # 1d data for volume confirmation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Volume confirmation: volume > 1.5x 20-period EMA on 1d
    vol_ema20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema20_1d)
    vol_confirm = volume > (1.5 * vol_ema20_1d_aligned)
    
    # Trend filter: close > 1d EMA50 for long, close < 1d EMA50 for short
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema20_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation + uptrend (close > EMA50)
            if price > highest_high[i] and vol_confirm[i] and price > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume confirmation + downtrend (close < EMA50)
            elif price < lowest_low[i] and vol_confirm[i] and price < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if price < midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if price > midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals