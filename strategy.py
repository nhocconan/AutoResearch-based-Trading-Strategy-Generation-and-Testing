#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(50) trend + volume spike filter
# Donchian breakouts capture momentum; 1d HMA ensures higher timeframe alignment
# Volume spike (>2x avg) confirms institutional participation; discrete sizing 0.25
# Works in bull/bear: trend filter adapts, breakouts work in both directions
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing

name = "4h_1d_donchian_hma_volume_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d HMA(50)
    close_1d = df_1d['close'].values
    half_len = 50 // 2
    sqrt_len = int(np.sqrt(50))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        wma_vals = np.full(len(values), np.nan)
        for i in range(window - 1, len(values)):
            wma_vals[i] = np.dot(values[i - window + 1:i + 1], weights) / weights.sum()
        return wma_vals
    
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 50)
    hma_1d = 2 * wma_half - wma_full
    hma_1d = wma(hma_1d, sqrt_len)
    
    # Align 1d HMA to 4h timeframe (wait for 1d bar close)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume spike filter
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(hma_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2.0x 20-period average
        volume_spike = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < 1d HMA (trend change)
            if close[i] < donchian_low[i] or close[i] < hma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > 1d HMA (trend change)
            if close[i] > donchian_high[i] or close[i] > hma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume spike and Donchian breakout + 1d HMA filter
            if volume_spike:
                # Long entry: price > Donchian high AND price > 1d HMA (bullish alignment)
                if close[i] > donchian_high[i] and close[i] > hma_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND price < 1d HMA (bearish alignment)
                elif close[i] < donchian_low[i] and close[i] < hma_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals