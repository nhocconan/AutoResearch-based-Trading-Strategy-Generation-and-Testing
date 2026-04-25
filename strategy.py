#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm
Hypothesis: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1d trend filter (price >/<- EMA50) and volume confirmation (>1.5x 20-bar avg). Enters long when Bull Power > 0 and rising in 1d uptrend, short when Bear Power > 0 and rising in 1d downtrend. Uses discrete sizing (0.25) to limit fee churn. Designed for 6h timeframe with ~12-30 trades/year, works in bull/bear by following 1d trend filter.
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
    
    # Elder Ray components (EMA13 for 6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # Slope of Bull/Bear Power (rising if current > previous)
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = bull_power[0]
    bear_power_prev[0] = bear_power[0]
    bull_power_rising = bull_power > bull_power_prev
    bear_power_rising = bear_power > bear_power_prev
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 13-period data for EMA13 and 50 for 1d EMA
    start_idx = max(13, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising in 1d uptrend with volume confirmation
            long_setup = (bull_power[i] > 0) and bull_power_rising[i] and (close_1d[i] > ema_50_1d_aligned[i]) and volume_spike[i]
            # Short: Bear Power > 0 and rising in 1d downtrend with volume confirmation
            short_setup = (bear_power[i] > 0) and bear_power_rising[i] and (close_1d[i] < ema_50_1d_aligned[i]) and volume_spike[i]
            
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
            # Exit: Bull Power <= 0 OR trend turns down
            if (bull_power[i] <= 0) or (close_1d[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bear Power <= 0 OR trend turns up
            if (bear_power[i] <= 0) or (close_1d[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0