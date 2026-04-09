#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(50) trend + volume confirmation
# Donchian captures breakouts; 1d HMA confirms higher timeframe trend direction
# Volume ensures breakout authenticity; discrete sizing 0.20 limits drawdown
# Session filter (08-20 UTC) reduces noise trades
# Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing
# Works in bull/bear: trend filter adapts, breakouts work in both directions

name = "4h_1d_donchian_hma_volume_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = prices.index.hour
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high_4h = np.full(len(close_4h), np.nan)
    donchian_low_4h = np.full(len(close_4h), np.nan)
    
    for i in range(len(close_4h)):
        if i < 20:
            donchian_high_4h[i] = np.nan
            donchian_low_4h[i] = np.nan
        else:
            donchian_high_4h[i] = np.max(high_4h[i-20:i])
            donchian_low_4h[i] = np.min(low_4h[i-20:i])
    
    # Align 4h Donchian to 1h timeframe (wait for 4h bar close)
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
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
    
    # Align 1d HMA to 1h timeframe (wait for 1d bar close)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
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
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(hma_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < 4h Donchian low OR price < 1d HMA (trend change)
            if close[i] < donchian_low_4h_aligned[i] or close[i] < hma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price > 4h Donchian high OR price > 1d HMA (trend change)
            if close[i] > donchian_high_4h_aligned[i] or close[i] > hma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic with volume confirmation and 4h Donchian breakout + 1d HMA filter
            if volume_confirmed:
                # Long entry: price > 4h Donchian high AND price > 1d HMA (bullish alignment)
                if close[i] > donchian_high_4h_aligned[i] and close[i] > hma_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: price < 4h Donchian low AND price < 1d HMA (bearish alignment)
                elif close[i] < donchian_low_4h_aligned[i] and close[i] < hma_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals