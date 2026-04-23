#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power + 1d EMA34 trend filter + volume spike.
- Primary timeframe: 6h, HTF: 1d for trend filter and Elder Ray calculation
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close)
- Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period avg
- Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period avg
- Exit: Elder Ray momentum weakens (Bull Power <= 0 for longs, Bear Power >= 0 for shorts)
- Uses Elder Ray to measure bull/bear power relative to EMA13 for momentum confirmation
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to balance return and risk
- BTC/ETH focus: requires 1d EMA34 trend alignment to avoid SOL-only bias
- Works in bull markets (strong bull power with uptrend) and bear markets (strong bear power with downtrend)
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
    
    # Volume confirmation: > 2.0x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d
    bear_power = low - ema_13_1d
    
    # Align Elder Ray to 6h timeframe (values from previous 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA34 (uptrend) AND volume spike
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                close[i] > ema_34_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 1d EMA34 (downtrend) AND volume spike
            elif (bull_power_aligned[i] < 0 and 
                  bear_power_aligned[i] > 0 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Elder Ray momentum weakens (Bull Power <= 0)
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Elder Ray momentum weakens (Bear Power >= 0)
            if bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0