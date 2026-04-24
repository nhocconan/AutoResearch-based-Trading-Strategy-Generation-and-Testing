#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Donchian(20) from 1w provides robust structure for 12h breakouts.
- 1w EMA50 trend filter ensures alignment with weekly momentum, reducing false signals in choppy markets.
- Volume spike (>2.0x 12-period average volume) confirms breakout validity with higher threshold to reduce whipsaws.
- Discrete position sizing (0.25) balances return and drawdown control.
- Target: 60-120 total trades over 4 years (15-30/year) on 12h timeframe.
- Uses 1w HTF data loaded ONCE before loop per MTF rules.
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
    
    # Get 1w data ONCE before loop for Donchian and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Donchian(20) levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper (20-period high) and lower (20-period low)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (using previous completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2.0x 12-period average volume (12h * 1 = half day)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 12) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike and above 1w EMA50
            if close[i] > donchian_high_aligned[i] and volume_spike[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and below 1w EMA50
            elif close[i] < donchian_low_aligned[i] and volume_spike[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low OR below 1w EMA50
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high OR above 1w EMA50
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0