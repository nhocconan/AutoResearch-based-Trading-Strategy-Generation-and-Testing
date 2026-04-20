#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 4x4 Grid Trading with Daily Trend Filter
# - Buy when price breaks above 4x4 grid upper resistance (20-period high)
# - Sell when price breaks below 4x4 grid lower support (20-period low)
# - Only take long when price > 1d 200-day EMA, short when price < 1d 200-day EMA
# - Grid provides clear support/resistance levels based on recent price action
# - Daily EMA filter ensures alignment with higher timeframe trend
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 15-35 trades per year per symbol (60-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-day EMA on 1d timeframe
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4x4 grid levels on 4h timeframe
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Upper resistance: 20-period high
    grid_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower support: 20-period low
    grid_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(grid_upper[i]) or np.isnan(grid_lower[i]) or np.isnan(ema_200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine price position relative to grid and 1d EMA
        price_above_grid = close_4h[i] > grid_upper[i]
        price_below_grid = close_4h[i] < grid_lower[i]
        price_above_ema = close_4h[i] > ema_200_1d_aligned[i]
        price_below_ema = close_4h[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above grid + above 1d EMA
            if price_above_grid and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below grid + below 1d EMA
            elif price_below_grid and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below grid OR falls below 1d EMA
            if price_below_grid or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above grid OR rises above 1d EMA
            if price_above_grid or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4x4_Grid_1dEMA200_Filter"
timeframe = "4h"
leverage = 1.0