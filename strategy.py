#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA13 for Elder Ray and trend
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = ema13_1d - df_1d['low'].values
    
    # Align Elder Ray and trend to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    trend_1d = (close_1d > ema13_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0 (both positive), uptrend, volume spike
            long_cond = (bull_power_6h[i] > 0 and bear_power_6h[i] > 0 and 
                        trend_1d_aligned[i] > 0.5 and vol_spike[i])
            
            # Short: Bull Power < 0, Bear Power < 0 (both negative), downtrend, volume spike
            short_cond = (bull_power_6h[i] < 0 and bear_power_6h[i] < 0 and 
                         trend_1d_aligned[i] < 0.5 and vol_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative or trend turns down
            if bull_power_6h[i] <= 0 or trend_1d_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns negative or trend turns up
            if bear_power_6h[i] <= 0 or trend_1d_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) identifies strong momentum with volume spike confirmation.
# Long when both powers positive (strong uptrend), short when both negative (strong downtrend).
# Uses 1d EMA13 trend filter for multi-timeframe alignment. Target: 20-60 trades/year.
# Works in both bull/bear markets by capturing strong directional moves with volume confirmation.