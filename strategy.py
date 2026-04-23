#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 20-day high AND close > 1w EMA50 (uptrend) AND volume > 1.5x 20-day MA.
Short when price breaks below 20-day low AND close < 1w EMA50 (downtrend) AND volume > 1.5x 20-day MA.
Exit when price returns to opposite Donchian level or opposite extreme is hit.
Designed for ~10-20 trades/year with structure-based edge in trending markets.
Donchian channels provide clear breakout levels; 1w EMA50 ensures higher timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1w EMA50 = uptrend, close < 1w EMA50 = downtrend
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter: 1d volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i]  # Break above 20-day high
        breakout_down = close[i] < low_20[i]  # Break below 20-day low
        return_to_low = close[i] < low_20[i]  # Return below 20-day low (exit long)
        return_to_high = close[i] > high_20[i]  # Return above 20-day high (exit short)
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above 20-day high AND uptrend AND volume confirmation
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-day low AND downtrend AND volume confirmation
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to opposite Donchian level or opposite extreme hit
            exit_signal = False
            if position == 1:
                exit_signal = return_to_low or opposite_extreme
            elif position == -1:
                exit_signal = return_to_high or opposite_extreme
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0