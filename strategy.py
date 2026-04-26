#!/usr/bin/env python3
"""
6h_ElderRay_ZeroCross_1dTrend_VolumeSpike
Hypothesis: Use 6h timeframe with Elder Ray (Bull/Bear Power) zero cross, confirmed by 1d EMA50 trend and volume spike.
Long when: Bull Power crosses above zero + Bear Power < 0 + 1d EMA50 uptrend + volume > 1.5 * avg volume.
Short when: Bear Power crosses below zero + Bull Power > 0 + 1d EMA50 downtrend + volume > 1.5 * avg volume.
Exit when: Elder Ray power of opposite sign crosses zero (reversal) or volume dries up.
Uses discrete 0.25 position size to limit fee drag. Designed for BTC/ETH:
- Elder Ray measures bull/bear power relative to EMA13, effective in trending markets
- 1d EMA50 filter ensures trading with the daily trend
- Volume confirmation avoids low-validity signals
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
    
    # Calculate Elder Ray (Bull Power, Bear Power) using EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 13 for EMA13, 20 for volume avg, 50 for 1d EMA
    start_idx = max(13, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Previous values for crossover detection
        bull_prev = bull_power[i-1]
        bear_prev = bear_power[i-1]
        bull_curr = bull_power[i]
        bear_curr = bear_power[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for zero cross with trend and volume confirmation
            # Long: Bull Power crosses above zero + Bear Power < 0 + 1d EMA50 uptrend + volume spike
            long_entry = (bull_prev <= 0 and bull_curr > 0) and \
                       (bear_curr < 0) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_spike[i]
            # Short: Bear Power crosses below zero + Bull Power > 0 + 1d EMA50 downtrend + volume spike
            short_entry = (bear_prev >= 0 and bear_curr < 0) and \
                        (bull_curr > 0) and \
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
            # Long - exit when Bear Power crosses above zero (reversal) or volume dries up
            if (bear_prev < 0 and bear_curr >= 0) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when Bull Power crosses above zero (reversal) or volume dries up
            if (bull_prev <= 0 and bull_curr > 0) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_ZeroCross_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0