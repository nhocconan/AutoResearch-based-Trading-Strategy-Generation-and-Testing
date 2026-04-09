#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
# Donchian captures breakouts; 12h HMA confirms higher timeframe trend direction
# Volume ensures breakout authenticity; discrete sizing 0.30 limits drawdown
# Works in bull/bear: trend filter adapts, breakouts work in both directions
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing

name = "4h_12h_donchian_hma_volume_v1"
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
    
    # Load 12h data ONCE before loop for HMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h HMA(21)
    close_12h = df_12h['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        wma_vals = np.full(len(values), np.nan)
        for i in range(window - 1, len(values)):
            wma_vals[i] = np.dot(values[i - window + 1:i + 1], weights) / weights.sum()
        return wma_vals
    
    wma_half = wma(close_12h, half_len)
    wma_full = wma(close_12h, 21)
    hma_12h = 2 * wma_half - wma_full
    hma_12h = wma(hma_12h, sqrt_len)
    
    # Align 12h HMA to 4h timeframe (wait for 12h bar close)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
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
    
    # Calculate 20-period average volume for volume confirmation
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
            np.isnan(hma_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < 12h HMA (trend change)
            if close[i] < donchian_low[i] or close[i] < hma_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > 12h HMA (trend change)
            if close[i] > donchian_high[i] or close[i] > hma_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + 12h HMA filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND price > 12h HMA (bullish alignment)
                if close[i] > donchian_high[i] and close[i] > hma_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Short entry: price < Donchian low AND price < 12h HMA (bearish alignment)
                elif close[i] < donchian_low[i] and close[i] < hma_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals