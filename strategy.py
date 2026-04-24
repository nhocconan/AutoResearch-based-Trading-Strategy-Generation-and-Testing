#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Uses Donchian channel (20-period high/low) from prior completed 12h candles.
- Breakout above upper channel or below lower channel with volume > 2.0x 20-bar average signals strong momentum.
- Regime filter: ATR(14) ratio (current ATR / 50-period ATR) < 0.8 indicates low volatility environment favorable for breakouts.
- Designed for 12h timeframe to capture medium-term breakouts with lower trade frequency.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from prior completed 12h candles
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper and lower Donchian channels (20-period lookback)
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe (wait for 12h bar to close)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR ratio: current ATR(14) / ATR(50) - values < 0.8 indicate low volatility
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 1.0)
    
    # Align ATR ratio to 12h timeframe (wait for 1d bar to close)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # ATR regime filter: low volatility environment (ATR ratio < 0.8)
        regime_filter = atr_ratio_aligned[i] < 0.8
        
        if position == 0:
            # Long: breakout above upper Donchian AND volume confirmation AND regime filter
            if close[i] > upper_12h_aligned[i] and volume_confirm and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower Donchian AND volume confirmation AND regime filter
            elif close[i] < lower_12h_aligned[i] and volume_confirm and regime_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below lower Donchian OR ATR regime changes to high volatility
            if close[i] < lower_12h_aligned[i] or atr_ratio_aligned[i] >= 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above upper Donchian OR ATR regime changes to high volatility
            if close[i] > upper_12h_aligned[i] or atr_ratio_aligned[i] >= 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0