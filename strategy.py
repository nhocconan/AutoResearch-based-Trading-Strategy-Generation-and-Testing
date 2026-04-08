#!/usr/bin/env python3
"""
4x4 Grid Strategy with 1d Trend Filter
Hypothesis: A 4x4 grid of support/resistance levels based on 1d high/low and ATR provides
clear entry zones. Combined with 1d EMA trend filter and volume confirmation, this
captures breakouts and mean-reversion within ranges. Grid structure reduces whipsaw.
Designed to work in bull via breakouts, in bear via mean-reversion to grid levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4x4_grid_strategy"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for grid and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for grid spacing
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4x4 grid levels: 4 above, 4 below 1d midpoint
    # Grid uses 1d ATR as unit size
    grid_unit = atr_14 * 0.5  # Half ATR spacing for tighter grid
    midpoint = (high_1d + low_1d) / 2
    
    # Generate 4 levels above and below midpoint
    grid_upper = np.zeros_like(midpoint)
    grid_lower = np.zeros_like(midpoint)
    for i in range(1, 5):
        grid_upper += grid_unit * i
        grid_lower -= grid_unit * i
    
    # Align grid levels to 4h
    grid_upper_aligned = [align_htf_to_ltf(prices, df_1d, grid_upper + midpoint) for _ in range(4)]
    grid_lower_aligned = [align_htf_to_ltf(prices, df_1d, grid_lower + midpoint) for _ in range(4)]
    
    # Flatten grid levels for easy access
    grid_levels = []
    for i in range(4):
        grid_levels.append(grid_lower_aligned[i])
        grid_levels.append(grid_upper_aligned[i])
    grid_levels = np.array(grid_levels)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike[i]):
            signals[i] = 0.0
            continue
        
        # Skip if grid levels not ready
        if any(np.isnan(grid_levels[:, i])) if grid_levels.ndim > 1 else False:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below nearest grid level or trend reverses
            if close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above nearest grid level or trend reverses
            if close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Check if price is near any grid level (within 0.2 * ATR)
            near_grid = False
            for level in grid_levels:
                if not np.isnan(level[i]) and abs(close[i] - level[i]) < (atr_14[i] * 0.2 if not np.isnan(atr_14[i]) else np.inf):
                    near_grid = True
                    break
            
            # Long: price above grid + uptrend + volume spike
            if uptrend and vol_spike[i] and near_grid:
                position = 1
                signals[i] = 0.25
            # Short: price below grid + downtrend + volume spike
            elif downtrend and vol_spike[i] and near_grid:
                position = -1
                signals[i] = -0.25
    
    return signals