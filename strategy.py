#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) Breakout with 1d ATR Trend Filter and Volume Confirmation.
- Donchian(20) from 12h chart captures medium-term momentum breakouts.
- 1d ATR(14) trend filter: ATR rising = increasing volatility (trend), ATR falling = decreasing volatility (range).
- Volume confirmation: >1.8x 24-period average volume validates breakout strength.
- Discrete position sizing (0.25) minimizes fee churn.
- Target trades: 50-150 total over 4 years (12-37/year) on 12h timeframe.
- Works in bull/bear markets via volatility-based trend filter and volume confirmation.
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
    
    # Get 1d data ONCE before loop for ATR trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ATR(14) trend filter: ATR rising = trend, ATR falling = range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR trend: rising ATR = trending market (use breakout), falling ATR = ranging (avoid)
    atr_rising = atr_14_1d > np.roll(atr_14_1d, 1)
    atr_rising[0] = False
    
    # Align ATR trend to 12h timeframe (using previous completed 1d bar)
    atr_rising_aligned = align_htf_to_ltf(prices, df_1d, atr_rising)
    
    # Donchian(20) from 12h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.8x 24-period average volume (12h * 2 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 24, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_rising_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike and ATR rising (trending market)
            if close[i] > donchian_high[i] and volume_spike[i] and atr_rising_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and ATR rising (trending market)
            elif close[i] < donchian_low[i] and volume_spike[i] and atr_rising_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low OR ATR falling (range developing)
            if close[i] < donchian_low[i] or not atr_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high OR ATR falling (range developing)
            if close[i] > donchian_high[i] or not atr_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATRTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0