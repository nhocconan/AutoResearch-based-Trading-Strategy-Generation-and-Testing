#!/usr/bin/env python3
# 6h_ElderRay_Energy_1dTrend_Volume
# Hypothesis: Uses Elder Ray (Bull/Bear Power) to measure buying/selling pressure, filtered by daily trend and volume spikes.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Enter long when Bull Power > 0 and rising, Bear Power < 0 and falling, with volume confirmation.
# Works in bull/bear by aligning with daily EMA34 trend to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

name = "6h_ElderRay_Energy_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Slope of Bull/Bear Power (3-period change)
    bull_power_slope = bull_power - np.roll(bull_power, 3)
    bear_power_slope = bear_power - np.roll(bear_power, 3)
    bull_power_slope[:3] = 0
    bear_power_slope[:3] = 0
    
    # Get daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20, 3)  # Warmup for EMA13, volume MA, and slope
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: Bull Power > 0 and rising, Bear Power < 0, with volume confirmation and uptrend
            if (bull_power[i] > 0 and bull_power_slope[i] > 0 and 
                bear_power[i] < 0 and volume_confirm and uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 and rising, Bull Power < 0, with volume confirmation and downtrend
            elif (bear_power[i] > 0 and bear_power_slope[i] > 0 and 
                  bull_power[i] < 0 and volume_confirm and downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative or trend turns down
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns negative or trend turns up
            if bear_power[i] <= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals