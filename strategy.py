#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
Long when price breaks above 20-period Donchian high AND 1d ATR ratio (current/20-period MA) > 1.2 AND volume > 1.5x 20-period average.
Short when price breaks below 20-period Donchian low AND 1d ATR ratio > 1.2 AND volume > 1.5x 20-period average.
Exit when price touches the opposite Donchian level.
Uses 1d HTF for volatility expansion filter (avoids low-momentum breakouts that fail). Target: 75-200 total trades over 4 years (19-50/year).
Donchian breakouts capture strong momentum moves; volatility filter ensures we trade during high-momentum regimes (works in both bull and bear markets when volatility expands).
"""

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
    
    # Calculate 1d ATR for volatility filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = np.where(atr_ma_1d > 0, atr_1d / atr_ma_1d, 1.0)
    
    # Align 1d ATR ratio to 4h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20 + 19)  # donchian (20), atr calculation (20+19)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_ratio = atr_ratio_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above Donchian high AND volatility expansion AND volume spike
            if price > upper and atr_ratio > 1.2 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND volatility expansion AND volume spike
            elif price < lower and atr_ratio > 1.2 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Donchian level
            if position == 1 and price < lower:  # Long exit at Donchian low
                exit_signal = True
            elif position == -1 and price > upper:  # Short exit at Donchian high
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATRratio_VolumeConfirmation_LevelExit"
timeframe = "4h"
leverage = 1.0