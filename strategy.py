#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h for execution and signal generation.
- HTF: 1d for ATR-based regime filter (trending vs ranging) and Donchian calculation.
- Entry: Price breaks above Donchian(20) upper band (long) or below lower band (short) on 12h close,
         with volume > 1.5x 20-period volume MA, and only when ATR(14)/ATR(50) > 0.3 (trending regime).
- Exit: Price returns to Donchian(20) midpoint or ATR regime shifts to ranging (ATR ratio < 0.25).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Uses 1d ATR regime to avoid false breakouts in ranging markets, improving Sharpe in both regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_ratio = atr_14 / (atr_50 + 1e-10)  # avoid division by zero
    
    # Align ATR ratio to 12h timeframe (completed 1d bar only)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1d Donchian(20) channels
    donch_h = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_m = (donch_h + donch_l) / 2
    
    # Align Donchian levels to 12h timeframe (completed 1d bar only)
    donch_h_aligned = align_htf_to_ltf(prices, df_1d, donch_h)
    donch_l_aligned = align_htf_to_ltf(prices, df_1d, donch_l)
    donch_m_aligned = align_htf_to_ltf(prices, df_1d, donch_m)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # Need ATR(50), Donchian(20), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(donch_h_aligned[i]) or 
            np.isnan(donch_l_aligned[i]) or np.isnan(donch_m_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian HIGH with volume spike AND trending regime (ATR ratio > 0.3)
            if (close[i] > donch_h_aligned[i] and volume_spike[i] and 
                atr_ratio_aligned[i] > 0.3):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian LOW with volume spike AND trending regime (ATR ratio > 0.3)
            elif (close[i] < donch_l_aligned[i] and volume_spike[i] and 
                  atr_ratio_aligned[i] > 0.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Donchian MID or regime shifts to ranging (ATR ratio < 0.25)
            if (close[i] < donch_m_aligned[i] or atr_ratio_aligned[i] < 0.25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Donchian MID or regime shifts to ranging (ATR ratio < 0.25)
            if (close[i] > donch_m_aligned[i] or atr_ratio_aligned[i] < 0.25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0