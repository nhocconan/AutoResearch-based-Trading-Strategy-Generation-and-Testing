#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_1dTrend_VolumeSpike
Hypothesis: Elder Ray (Bull/Bear Power) zero cross signals combined with 1d EMA50 trend filter and volume confirmation.
Long when: Bear Power crosses above zero (bulls taking control) + 1d EMA50 uptrend + volume > 1.5 * avg volume.
Short when: Bull Power crosses below zero (bears taking control) + 1d EMA50 downtrend + volume > 1.5 * avg volume.
Exit when: Opposite Elder Ray power crosses zero (reversal signal).
Uses discrete 0.25 position size to limit fee drag. Designed for BTC/ETH:
- Elder Ray zero cross detects momentum shifts early with less lag than MACD
- 1d EMA50 ensures trading with higher timeframe trend
- Volume confirmation filters low-validity signals
- Targets 12-37 trades/year for optimal test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (standard setting)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13   # Bulls' strength: ability to push price above EMA
    bear_power = low - ema13    # Bears' strength: ability to push price below EMA
    
    # Zero cross signals
    bull_power_cross_above = (bull_power > 0) & (bull_power <= 0)  # Previous <=0, current >0
    bear_power_cross_below = (bear_power < 0) & (bear_power >= 0)  # Previous >=0, current <0
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 13 for EMA13, 50 for 1d EMA, 20 for volume avg
    start_idx = max(13, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            i == 0):  # Need previous bar for cross detection
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for zero cross with trend and volume confirmation
            # Long: Bear Power crosses above zero + 1d EMA50 uptrend + volume spike
            long_entry = (bear_power[i] > 0 and bear_power[i-1] <= 0) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_spike[i]
            # Short: Bull Power crosses below zero + 1d EMA50 downtrend + volume spike
            short_entry = (bull_power[i] < 0 and bull_power[i-1] >= 0) and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when Bull Power crosses below zero (bears taking control)
            if bull_power[i] < 0 and bull_power[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when Bear Power crosses above zero (bulls taking control)
            if bear_power[i] > 0 and bear_power[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_ZeroCross_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0