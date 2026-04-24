#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Uses Donchian channel (20-period high/low) from prior completed 4h candles for structure.
- Breakout above upper band or below lower band with volume > 2.0x 20-bar average signals strong momentum.
- Regime filter: 1d ATR(14) / ATR(50) ratio > 1.2 indicates high volatility regime (favors breakouts).
- Designed for 4h timeframe to capture medium-term breakouts in both trending and high-volatility regimes.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-50 trades/year (75-200 total over 4 years) to stay fee-efficient.
- Based on proven pattern: Donchian breakout + volume confirmation showed strong performance in DB.
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
    
    # Get 4h data ONCE before loop for Donchian channels (using prior completed 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper band (20-period high)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band (20-period low)
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe (wait for 4h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.abs(high_1d[0] - low_1d[0])  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar (no prev close)
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # first bar (no prev close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: ratio > 1.2 indicates high volatility (breakout-favorable)
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # ATR regime filter: high volatility regime (ratio > 1.2)
        high_vol_regime = atr_ratio_aligned[i] > 1.2
        
        if position == 0:
            # Long: breakout above Donchian high AND volume confirmation AND high vol regime
            if close[i] > donchian_high_aligned[i] and volume_confirm and high_vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND volume confirmation AND high vol regime
            elif close[i] < donchian_low_aligned[i] and volume_confirm and high_vol_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low OR low volatility regime
            if close[i] < donchian_low_aligned[i] or not high_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Donchian high OR low volatility regime
            if close[i] > donchian_high_aligned[i] or not high_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0