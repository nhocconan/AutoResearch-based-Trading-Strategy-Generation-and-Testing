#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Uses Donchian channel (20-period high/low) from 12h timeframe as price structure.
- Breakout above upper band with volume > 2.0x 20-bar average = long signal.
- Breakdown below lower band with volume > 2.0x 20-bar average = short signal.
- Regime filter: 1d ATR(14) / ATR(50) > 0.7 indicates sufficient volatility for breakout validity.
- Designed for 12h timeframe to capture multi-day trends with proper structure.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Combines proven Donchian breakout with ATR regime filter (better than EMA for volatility adaptation).
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
    
    # Get 12h data ONCE before loop for Donchian levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) for 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: 20-period high
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for 12h bar to close)
    upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: ATR(14)/ATR(50) > 0.7 indicates sufficient volatility
    atr_ratio = atr_14 / atr_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough for ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Regime filter: sufficient volatility
        volatile_regime = atr_ratio_aligned[i] > 0.7
        
        if position == 0:
            # Only trade if volume confirms breakout AND we're in volatile regime
            if volume_confirm and volatile_regime:
                # Long: price breaks above upper Donchian band
                if close[i] > upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower Donchian band
                elif close[i] < lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below lower band OR regime becomes unfavorable
            if close[i] < lower_aligned[i] or atr_ratio_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper band OR regime becomes unfavorable
            if close[i] > upper_aligned[i] or atr_ratio_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0