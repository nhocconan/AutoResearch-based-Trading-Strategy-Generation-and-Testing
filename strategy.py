#!/usr/bin/env python3
name = "1h_4h_1d_Trend_Filter_Breakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # 4h EMA(50) for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data once for breakout levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) on 1d for breakout levels (previous day)
    # Donchian high: highest high over last 20 days
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over last 20 days
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h (wait for daily close)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Volume spike: current volume > 1.5x 24-period average (1 day of 1h bars)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + 4h trend up + volume spike
            if (close[i] > donch_high_aligned[i] and 
                close[i] > ema50_4h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low + 4h trend down + volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals