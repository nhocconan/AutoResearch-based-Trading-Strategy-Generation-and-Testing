#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
Long when price breaks above Donchian upper (20) AND 1d ATR(14) > 1.5x 50-period MA(ATR) AND volume > 2.0x 20-period MA.
Short when price breaks below Donchian lower (20) AND 1d ATR(14) > 1.5x 50-period MA(ATR) AND volume > 2.0x 20-period MA.
Exit when price returns to Donchian midpoint or opposite breakout occurs.
Designed for ~20-30 trades/year with volatility-based edge that works in both trending and ranging markets.
ATR filter ensures we only trade during sufficient volatility regimes, avoiding choppy low-vol periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_14_1d > 1.5 * atr_ma_50_1d
    
    # Align ATR filter to 4h timeframe
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need ATR MA50, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_filter_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: 1d ATR > 1.5x 50-period MA(ATR)
        vol_filter = atr_filter_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA
        vol_confirm = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i]  # Break above upper band
        breakout_down = close[i] < donchian_lower[i]  # Break below lower band
        return_to_middle = abs(close[i] - donchian_middle[i]) < 0.1 * abs(donchian_upper[i] - donchian_lower[i])
        opposite_breakout = (position == 1 and breakout_down) or \
                            (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Break above upper band AND volatility filter AND volume confirmation
            if breakout_up and vol_filter and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band AND volatility filter AND volume confirmation
            elif breakout_down and vol_filter and vol_confirm:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to middle or opposite breakout
            exit_signal = False
            if position == 1:
                exit_signal = return_to_middle or opposite_breakout
            elif position == -1:
                exit_signal = return_to_middle or opposite_breakout
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATRFilter_VolumeConfirm"
timeframe = "4h"
leverage = 1.0