#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
- Long when price breaks above Donchian upper band (20-period high) AND 1d ATR(14) > 1d ATR(50) (expanding volatility)
- Short when price breaks below Donchian lower band (20-period low) AND 1d ATR(14) > 1d ATR(50) (expanding volatility)
- Volume confirmation: current volume > 1.5 * 20-period average volume (moderate spike to avoid overtrading)
- Exit on opposite Donchian breakout (lower band for long exit, upper band for short exit)
- Uses 12h primary with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Donchian channels provide clear breakout levels; ATR filter ensures volatility expansion; volume confirms momentum
- Designed to capture strong moves in both bull (breakouts up) and bear (breakouts down) markets
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate Donchian channels using previous 20-period high/low (avoid look-ahead)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for volatility filter
    df_1d_copy = df_1d.copy()
    df_1d_copy['high_low'] = df_1d_copy['high'] - df_1d_copy['low']
    df_1d_copy['high_prev_close'] = abs(df_1d_copy['high'] - df_1d_copy['close'].shift(1))
    df_1d_copy['low_prev_close'] = abs(df_1d_copy['low'] - df_1d_copy['close'].shift(1))
    df_1d_copy['true_range'] = df_1d_copy[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)
    
    atr_14 = df_1d_copy['true_range'].rolling(window=14, min_periods=14).mean().values
    atr_50 = df_1d_copy['true_range'].rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 12h timeframe (waits for completed 1d bar)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Volatility filter: ATR(14) > ATR(50) indicates expanding volatility
    vol_expanding = atr_14_aligned > atr_50_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average (moderate spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need ATR50 and Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian upper AND volatility expanding AND volume confirmation
            if close[i] > donchian_upper[i] and vol_expanding[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian lower AND volatility expanding AND volume confirmation
            elif close[i] < donchian_lower[i] and vol_expanding[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian lower (opposite band)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian upper (opposite band)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0