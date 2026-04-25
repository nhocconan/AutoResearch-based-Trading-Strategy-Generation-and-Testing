#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm
Hypothesis: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) with 1d trend filter (price >/< EMA50) and volume confirmation (>2.0x 20-bar avg). 
Enters long when Bull Power > 0 and rising (bullish momentum) in 1d uptrend with volume spike, short when Bear Power < 0 and falling (bearish momentum) in 1d downtrend with volume spike. 
Exits on opposite Elder Ray signal (Bear Power > 0 for longs exit, Bull Power < 0 for shorts exit) or trend reversal. 
Designed for 6h timeframe with ~15-35 trades/year, works in bull/bear by following 1d trend filter and momentum confirmation.
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
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray calculations on 6h timeframe
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 1 bar of previous data and EMA13/EMA50 warmup
    start_idx = max(13, 50, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND rising (bullish momentum) in 1d uptrend with volume confirmation
            bull_power_rising = bull_power[i] > bull_power[i-1]
            long_setup = (bull_power[i] > 0) and bull_power_rising and (close[i] > ema_50_1d_aligned[i]) and volume_spike[i]
            # Short: Bear Power < 0 AND falling (bearish momentum) in 1d downtrend with volume confirmation
            bear_power_falling = bear_power[i] < bear_power[i-1]
            short_setup = (bear_power[i] < 0) and bear_power_falling and (close[i] < ema_50_1d_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bear Power > 0 (momentum shift) OR trend turns down
            if (bear_power[i] > 0) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bull Power < 0 (momentum shift) OR trend turns up
            if (bull_power[i] < 0) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0