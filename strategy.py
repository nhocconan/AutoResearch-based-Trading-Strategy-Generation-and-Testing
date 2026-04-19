#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation
# Donchian(20) provides clear breakout signals in trending markets
# 12h EMA50 filters trades to align with higher timeframe trend
# Volume spike (>1.5x 20-period average) confirms breakout strength
# Target: 20-30 trades/year per symbol with disciplined entries
name = "4h_Donchian20_12hEMA_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA50 for trend bias
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close breaks above Donchian high + above 12h EMA + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below Donchian low + below 12h EMA + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if close breaks below Donchian low or below 12h EMA
            if (close[i] < donchian_low[i]) or (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if close breaks above Donchian high or above 12h EMA
            if (close[i] > donchian_high[i]) or (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals