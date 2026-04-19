# [6h] Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
# Hypothesis: Weekly pivot levels (from weekly chart) determine long-term trend direction.
# Breakouts above/below 6h Donchian(20) channels are traded only in direction of weekly pivot (above/below pivot).
# Volume confirmation filters breakouts with institutional participation.
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear via weekly pivot as trend filter and volume confirmation.
# Uses discrete position sizing (0.25) to minimize fee churn.

name = "6h_Donchian_20_Breakout_WeeklyPivot_Volume"
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
    
    # 6h Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly data for pivot direction (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot: (weekly high + weekly low + weekly close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly pivot aligned to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2)  # Ensure enough data for Donchian and weekly pivot
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly pivot trend filter
        above_pivot = close[i] > weekly_pivot_aligned[i]
        below_pivot = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot AND volume confirmation
            if (close[i] > donch_high[i] and 
                above_pivot and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below weekly pivot AND volume confirmation
            elif (close[i] < donch_low[i] and 
                  below_pivot and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low (reversal) or falls below weekly pivot
            if (close[i] < donch_low[i]) or (not above_pivot):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high (reversal) or rises above weekly pivot
            if (close[i] > donch_high[i]) or (not below_pivot):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals