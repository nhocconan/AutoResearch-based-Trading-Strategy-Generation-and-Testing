#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
- Uses Donchian channel breakout from 1d for structure (more reliable than Camarilla on 12h)
- 1d ATR(14) as volatility filter - only trade when volatility is elevated (avoid choppy periods)
- Volume > 2.0x 24-period average for confirmation (adjusts for 12h lower frequency)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
- Works in both bull/bear via volatility filter + volume confirmation
- Uses 1d HTF as specified in experiment parameters
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
    
    # Volume confirmation: > 2.0x 24-period average (adjusted for 12h lower frequency)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1d data for Donchian channel calculation (HTF as specified)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channel (20-period) from prior 1d bar
    # Upper = max(high, lookback=20)
    # Lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align Donchian levels to 1d timeframe (using completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # 1d data for ATR(14) volatility filter
    tr1 = pd.Series(high_1d - low_1d).values
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1))).values
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1))).values
    tr2[0] = 0  # first bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # ATR ratio: current ATR / 50-period average ATR (to detect elevated volatility)
    atr_ma_50 = pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14_1d_aligned / np.where(atr_ma_50 > 0, atr_ma_50, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 20, 14, 50)  # Volume MA, Donchian, ATR, ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Volatility filter: only trade when ATR ratio > 1.2 (elevated volatility)
        volatility_filter = atr_ratio[i] > 1.2
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper_aligned[i]  # Close above upper channel
        breakout_down = close[i] < donchian_lower_aligned[i]  # Close below lower channel
        
        if position == 0:
            # Long: Donchian upper breakout AND volume confirmation AND volatility filter
            if breakout_up and volume_confirm and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower breakdown AND volume confirmation AND volatility filter
            elif breakout_down and volume_confirm and volatility_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian lower breakdown OR volatility drops below threshold
            if breakout_down or atr_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian upper breakout OR volatility drops below threshold
            if breakout_up or atr_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATR_VolumeSpike_Filter_v1"
timeframe = "12h"
leverage = 1.0