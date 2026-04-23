#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Alligator: Jaw (EMA13,8), Teeth (EMA8,5), Lips (EMA5,3) - smoothed medians
- Long: Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA50 (uptrend) + volume > 1.8x 20-period avg
- Short: Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA50 (downtrend) + volume > 1.8x 20-period avg
- Exit: Alligator sleeping (jaws intertwined: |Lips-Jaw| < 0.1*ATR) OR opposite signal
- 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- ATR-based exit prevents whipsaws in ranging markets
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Works in bull (trend continuation via Alligator alignment) and bear (mean reversion via Alligator sleep)
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
    
    # Volume confirmation: > 1.8x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for exit condition (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator components (using medians with smoothing)
    # Jaw: 13-period SMMA smoothed 8 periods
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.rolling(window=8, min_periods=8).mean().values
    
    # Teeth: 8-period SMMA smoothed 5 periods
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean().values
    
    # Lips: 5-period SMMA smoothed 3 periods
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.rolling(window=3, min_periods=3).mean().values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 8, 5, 50)  # Need 20 for volume MA, 13/8/5 for Alligator, 50 for 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        # Alligator sleeping condition (jaws intertwined)
        alligator_sleep = np.abs(lips[i] - jaw[i]) < 0.1 * atr[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA50 (uptrend) + volume spike
            if volume_spike and lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA50 (downtrend) + volume spike
            elif volume_spike and lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator sleeping OR price < 1d EMA50 (trend break)
            if alligator_sleep or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator sleeping OR price > 1d EMA50 (trend break)
            if alligator_sleep or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0