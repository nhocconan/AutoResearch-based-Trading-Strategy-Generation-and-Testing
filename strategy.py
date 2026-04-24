#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ATR regime filter (trending vs ranging).
- Entry: Price breaks above Donchian(20) high (long) or below Donchian(20) low (short) on 4h close,
         with volume > 1.5x 20-period volume MA and 1d ATR ratio > 0.7 (trending regime).
- Direction filter: only long when 4h close > 4h EMA50, only short when 4h close < 4h EMA50.
- ATR regime filter: 1d ATR(14) / 1d ATR(50) > 0.7 indicates trending market (avoid false breakouts in ranges).
- Volume confirmation reduces false breakouts.
- Exit: Price returns to 4h EMA50 or opposite Donchian band touch.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Avoids overtrading by requiring confluence of breakout, volume, regime, and trend alignment.
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
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR ratio: ATR(14)/ATR(50) > 0.7 indicates trending regime
    atr_ratio = atr_14 / np.where(atr_50 == 0, 1, atr_50)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Align Donchian channels to 4h timeframe (completed 1d bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # Need EMA50, Donchian(20), volume MA(20), plus 1 for safety
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike AND trending regime AND uptrend (close > EMA50)
            if (close[i] > donchian_high_aligned[i] and volume_spike[i] and 
                atr_ratio_aligned[i] > 0.7 and close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume spike AND trending regime AND downtrend (close < EMA50)
            elif (close[i] < donchian_low_aligned[i] and volume_spike[i] and 
                  atr_ratio_aligned[i] > 0.7 and close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to EMA50 or touches opposite Donchian band
            if (close[i] < ema_50[i] or close[i] < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to EMA50 or touches opposite Donchian band
            if (close[i] > ema_50[i] or close[i] > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0