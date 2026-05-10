#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_VolumeSpike
# Hypothesis: Use Elder Ray (Bull/Bear Power) on 6h for momentum, filtered by 1d EMA trend and 6h volume spikes.
# Bull Power = High - EMA13; Bear Power = Low - EMA13. Enter long when Bull Power turns positive in uptrend with volume spike.
# Enter short when Bear Power turns negative in downtrend with volume spike. Works in bull/bear via trend filter.
# Target: 15-25 trades/year to minimize fee drag.

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 6h EMA13 for Elder Ray
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13_6h  # High - EMA13
    bear_power = low - ema13_6h   # Low - EMA13
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # volume MA20 and EMA13
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power turns positive (>0) in uptrend with volume spike
            if (bull_power[i] > 0 and bull_power[i-1] <= 0 and  # crossover zero
                trend_1d_up_aligned[i] > 0.5 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power turns negative (<0) in downtrend with volume spike
            elif (bear_power[i] < 0 and bear_power[i-1] >= 0 and  # crossover zero
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bear Power turns negative or volume dries up
            if (bear_power[i] < 0 and bear_power[i-1] >= 0) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bull Power turns positive or volume dries up
            if (bull_power[i] > 0 and bull_power[i-1] <= 0) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals