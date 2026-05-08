#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d Trend Filter and Volume Spike
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when Bull Power > 0 and rising, Bear Power < 0, with 1d uptrend + volume spike
# - Short when Bear Power < 0 and falling, Bull Power < 0, with 1d downtrend + volume spike
# - Uses EMA13 for power calculation and EMA50 for 1d trend filter
# - Designed to capture institutional buying/selling pressure with trend alignment
# - Target: 12-30 trades/year to minimize fee drag on 6h timeframe

name = "6h_ElderRay_1dTrend_Volume"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Smooth the power signals to reduce noise
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for EMA13 and smoothing
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive AND rising, Bear Power negative, 1d uptrend + volume spike
            long_cond = (bull_power_smooth[i] > 0 and 
                        bull_power_smooth[i] > bull_power_smooth[i-1] and
                        bear_power_smooth[i] < 0 and
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: Bear Power negative AND falling, Bull Power negative, 1d downtrend + volume spike
            short_cond = (bear_power_smooth[i] < 0 and 
                         bear_power_smooth[i] < bear_power_smooth[i-1] and
                         bull_power_smooth[i] < 0 and
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative or Bear Power turns positive
            if bull_power_smooth[i] <= 0 or bear_power_smooth[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive or Bull Power turns positive
            if bear_power_smooth[i] >= 0 or bull_power_smooth[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals