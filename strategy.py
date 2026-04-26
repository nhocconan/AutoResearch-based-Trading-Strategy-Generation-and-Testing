#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dRegime
Hypothesis: On 6h timeframe, Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure. 
Enter long when Bull Power > 0 AND 1d trend is up (close > EMA50) AND volume > 1.5x 20-period average.
Enter short when Bear Power < 0 AND 1d trend is down (close < EMA50) AND volume > 1.5x 20-period average.
Exit when Bull/Bear Power crosses zero OR 1d trend reverses. Uses discrete sizing (0.0, ±0.25).
Target: 12-37 trades/year. Works in both bull (Bull Power strong) and bear (Bear Power strong) regimes.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Using 13-period EMA on 6h for sensitivity
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13  # Bull Power: buying pressure
    bear_power = low - ema_13   # Bear Power: selling pressure (negative values)
    
    # Volume confirmation: fixed threshold of 1.5x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA13 and volume MA warmup
    start_idx = max(13, 20)  # EMA13 needs 13, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0
        bear_strong = bear_power[i] < 0
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power positive + volume spike + 1d uptrend
            long_signal = bull_strong and volume_spike[i] and trend_uptrend
            
            # Short: Bear Power negative + volume spike + 1d downtrend
            short_signal = bear_strong and volume_spike[i] and trend_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power turns negative OR 1d trend changes to downtrend
            if not bull_strong or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bear Power turns positive OR 1d trend changes to uptrend
            if not bear_strong or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dRegime"
timeframe = "6h"
leverage = 1.0