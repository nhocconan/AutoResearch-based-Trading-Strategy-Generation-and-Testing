#!/usr/bin/env python3
name = "6h_Donchian_20_12hTrend_Volume"
timeframe = "6h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    ema_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channel (20) on 6h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > Donchian high with 12h uptrend and volume
            if (close[i] > high_max[i] and close[i] > ema_12h_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian low with 12h downtrend and volume
            elif (close[i] < low_min[i] and close[i] < ema_12h_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < Donchian low or trend change
            if close[i] < low_min[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > Donchian high or trend change
            if close[i] > high_max[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation.
# Donchian breakouts capture momentum; 12h trend filter ensures alignment with higher timeframe bias.
# Volume filter confirms institutional participation. Works in bull/bear markets by only trading in trend direction.
# Target: 20-40 trades/year to minimize fee drag. Position size 0.25 limits drawdown.