#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
Long when price breaks above 20-period high AND 1d ATR(14) < 0.5 * 20-period ATR AND volume > 1.5x 20-period MA.
Short when price breaks below 20-period low AND 1d ATR(14) < 0.5 * 20-period ATR AND volume > 1.5x 20-period MA.
Exit when price returns to opposite Donchian level or ATR filter fails.
Designed for ~25-35 trades/year with structure-based edge in low-volatility breakouts.
ATR filter ensures breakouts occur during compression, reducing false signals.
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
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(20) for volatility filter
    tr_4h1 = high - low
    tr_4h2 = np.abs(high - np.roll(close, 1))
    tr_4h3 = np.abs(low - np.roll(close, 1))
    tr_4h1[0] = high[0] - low[0]
    tr_4h2[0] = 0
    tr_4h3[0] = 0
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_20_4h = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # need Donchian20, ATR20, ATR14_1d, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_20_4h[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: 1d ATR < 0.5 * 4h ATR (low volatility environment)
        vol_filter = atr_14_1d_aligned[i] < 0.5 * atr_20_4h[i]
        
        # Volume filter: 4h volume > 1.5x 20-period MA
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i]  # Break above 20-period high
        breakout_down = close[i] < lowest_20[i]  # Break below 20-period low
        return_to_low = close[i] < lowest_20[i] + 0.1 * (highest_20[i] - lowest_20[i])  # Exit long near lower band
        return_to_high = close[i] > highest_20[i] - 0.1 * (highest_20[i] - lowest_20[i])  # Exit short near upper band
        opposite_extreme = (position == 1 and breakout_down) or \
                           (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above upper band AND low volatility AND volume confirmation
            if breakout_up and vol_filter and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND low volatility AND volume confirmation
            elif breakout_down and vol_filter and vol_confirm:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to opposite band or extreme hit
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

name = "4H_Donchian20_Breakout_1dATR_Filter_VolumeConfirm"
timeframe = "4h"
leverage = 1.0