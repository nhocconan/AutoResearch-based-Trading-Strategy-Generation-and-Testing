#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume spike filter.
- Uses 4h Camarilla pivot levels (R4, S4) derived from previous 1d OHLC for breakout signals
- Long breakout: price > R4 + volume > 2.0x 20-period avg + price > 1d EMA50 (strong uptrend)
- Short breakdown: price < S4 + volume > 2.0x 20-period avg + price < 1d EMA50 (strong downtrend)
- Exit: price reverts to Camarilla pivot point (PP)
- Higher threshold R4/S4 (vs R3/S3) and volume spike 2.0x (vs 1.5x) to reduce trades and avoid fee drag
- 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
- Target: 20-50 trades/year (75-200 total over 4 years) to minimize fee drag on 4h timeframe
- Camarilla levels provide institutional reference points that work across bull/bear regimes
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
    
    # Volume confirmation: > 2.0x 20-period average (strong spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # Camarilla formulas:
    # PP = (high + low + close) / 3
    # R4 = PP + (high - low) * 1.1
    # S4 = PP - (high - low) * 1.1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    r4 = pp + (high_1d - low_1d) * 1.1
    s4 = pp - (high_1d - low_1d) * 1.1
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long breakout: price > R4 + volume spike + price > 1d EMA50 (strong uptrend)
            if volume_spike and close[i] > ema_50_aligned[i]:
                if close[i] > r4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short breakdown: price < S4 + volume spike + price < 1d EMA50 (strong downtrend)
            elif volume_spike and close[i] < ema_50_aligned[i]:
                if close[i] < s4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R4S4_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0