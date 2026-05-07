#!/usr/bin/env python3
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
    
    # Get 1d data for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 and EMA20 for Elder Ray
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA20
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema20_1d
    
    # Align Elder Ray components to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Trend filter: EMA20 slope (positive for uptrend)
    ema20_slope = np.diff(ema20_1d, prepend=ema20_1d[0])
    ema20_slope_aligned = align_htf_to_ltf(prices, df_1d, ema20_slope)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA20 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema20_slope_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, positive EMA20 slope, volume spike
            if bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and ema20_slope_aligned[i] > 0 and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, negative EMA20 slope, volume spike
            elif bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0 and ema20_slope_aligned[i] < 0 and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Elder Ray divergence or trend change
            if position == 1:
                if bull_power_aligned[i] < 0 or ema20_slope_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if bear_power_aligned[i] < 0 or ema20_slope_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals