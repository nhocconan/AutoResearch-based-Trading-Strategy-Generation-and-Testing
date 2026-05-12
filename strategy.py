#!/usr/bin/env python3
name = "1h_4h_Donchian_20_Breakout_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data once for trend and Donchian calculations
    df_4h = get_htf_data(prices, '4h')
    
    # 4h close for trend filter (EMA50)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h high/low for Donchian channel (period 20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 20-period rolling max/min on 4h data
    high_roll = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h (wait for 4h close)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, high_roll)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, low_roll)
    
    # Volume spike filter on 1h: current volume > 1.8x 24-period average
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
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
            # Long: price breaks above Donchian high + 4h uptrend + volume spike
            if (close[i] > donch_high_aligned[i] and 
                close[i] > ema50_4h_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low + 4h downtrend + volume spike
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