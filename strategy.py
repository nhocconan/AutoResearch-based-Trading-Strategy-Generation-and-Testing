#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1w EMA trend filter and volume confirmation
# Donchian captures breakouts with clear structure, 1w EMA filters counter-trend trades
# Volume confirmation ensures breakout strength, targeting 20-50 trades/year
name = "4h_Donchian20_1wEMA20_Volume"
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
    
    # 1w EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian period
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + above 1w EMA20 + volume confirmation
            if (close[i] > donchian_high[i-1] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low + below 1w EMA20 + volume confirmation
            elif (close[i] < donchian_low[i-1] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or falls below 1w EMA20
            if (close[i] < donchian_low[i]) or (close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or rises above 1w EMA20
            if (close[i] > donchian_high[i]) or (close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals