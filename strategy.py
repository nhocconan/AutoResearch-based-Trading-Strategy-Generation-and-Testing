#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume spike
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and rising + price > 1d EMA50 + volume spike
# Short when Bear Power > 0 and rising + price < 1d EMA50 + volume spike
# Works in both bull/bear by adapting to trend via EMA50 filter
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag
name = "6h_ElderRay_Trend_1dVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray components: EMA13 of close
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Smooth Elder Ray with 5-period EMA to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Rising condition: current > previous
    bull_rising = bull_power_smooth > np.roll(bull_power_smooth, 1)
    bear_rising = bear_power_smooth > np.roll(bear_power_smooth, 1)
    # Handle first element
    bull_rising[0] = False
    bear_rising[0] = False
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # after EMA13 and smoothing
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising + price > 1d EMA50 + volume spike
            long_cond = (bull_power_smooth[i] > 0) and \
                        bull_rising[i] and \
                        (close[i] > ema_50_1d_aligned[i]) and \
                        volume_spike[i]
            # Short: Bear Power > 0 and rising + price < 1d EMA50 + volume spike
            short_cond = (bear_power_smooth[i] > 0) and \
                         bear_rising[i] and \
                         (close[i] < ema_50_1d_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (momentum lost)
            if bull_power_smooth[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 (momentum lost)
            if bear_power_smooth[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals