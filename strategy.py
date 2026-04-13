#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
    # Long: price breaks above Donchian upper (20) AND volume > 1.5x 20-period average AND 12h HMA(21) rising
    # Short: price breaks below Donchian lower (20) AND volume > 1.5x 20-period average AND 12h HMA(21) falling
    # Exit: price returns to Donchian midpoint (mean reversion)
    # Using 12h for HMA trend (structure) and 4h only for entry timing and Donchian channels
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 20-50 trades/year (~80-200 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA(21) for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    close_12h = df_12h['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = np.array([wma(close_12h[i:i+21], half_len) if i+21 <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    wma_full = np.array([wma(close_12h[i:i+21], 21) if i+21 <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    wma_diff = 2 * wma_half - wma_full
    hma_12h = np.array([wma(wma_diff[i:i+len(wma_diff)], sqrt_len) if i+len(wma_diff) <= len(wma_diff) else np.nan 
                        for i in range(len(wma_diff))])
    
    # Align 12h HMA to 4h (wait for completed 12h bar)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # 4h Donchian channels (20-period)
    donch_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(donch_len, n):
        upper[i] = np.max(high[i-donch_len:i])
        lower[i] = np.min(low[i-donch_len:i])
        middle[i] = (upper[i] + lower[i]) / 2
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(hma_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if 12h HMA rising, only short if 12h HMA falling
        hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1]
        hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1]
        
        # Entry logic: Donchian breakout + volume + trend
        long_entry = (close[i] > upper[i]) and vol_confirm and hma_rising
        short_entry = (close[i] < lower[i]) and vol_confirm and hma_falling
        
        # Exit logic: return to Donchian midpoint (mean reversion)
        long_exit = close[i] < middle[i]
        short_exit = close[i] > middle[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0