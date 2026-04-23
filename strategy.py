#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation
- Long when price breaks above 12h Donchian upper (20-period) AND 1d ATR ratio > 1.2 AND volume > 1.5x 20-period average
- Short when price breaks below 12h Donchian lower (20-period) AND 1d ATR ratio > 1.2 AND volume > 1.5x 20-period average
- Exit when price crosses 12h Donchian middle (mean reversion) OR ATR ratio drops below 0.8 (volatility collapse)
- Uses 1d ATR ratio (current ATR / 20-period ATR) to filter for sufficient volatility - avoids ranging markets
- Volume confirmation reduces false breakouts
- Designed for both bull and bear markets: volatility filter ensures we only trade when there's enough momentum
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for ATR filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR and 20-period ATR for ratio
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = tr.rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 34)  # Need 20 for Donchian, 20 for volume, 34 for ATR ratio
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Volatility filter (using 1d ATR ratio)
        vol_ok = atr_ratio_1d_aligned[i] > 1.2  # Sufficient volatility
        vol_collapse = atr_ratio_1d_aligned[i] < 0.8  # Volatility collapse for exit
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + volatility OK + volume confirmation
            if breakout_up and vol_ok and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volatility OK + volume confirmation
            elif breakout_down and vol_ok and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Donchian middle OR volatility collapse
                if close[i] < donchian_middle[i] or vol_collapse:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above Donchian middle OR volatility collapse
                if close[i] > donchian_middle[i] or vol_collapse:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0