#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13 to detect trend strength
# Long when Bull Power > 0 and Bear Power < 0 (strong uptrend) + volume spike + price above 1d EMA34
# Short when Bear Power < 0 and Bull Power > 0 (strong downtrend) + volume spike + price below 1d EMA34
# Uses 1d timeframe for trend filter and EMA13 calculation to avoid overtrading on 6h
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
# Works in both bull and bear markets by following the 1d trend direction and using Elder Ray for momentum

name = "6h_ElderRay_BullBearPower_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 (Elder Ray) and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray calculation
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Elder Ray signals with 1d trend filter
        # Long: Bull Power > 0 AND Bear Power < 0 (strong uptrend) + volume spike + price above 1d EMA34
        # Short: Bear Power < 0 AND Bull Power > 0 (strong downtrend) + volume spike + price below 1d EMA34
        if position == 0:
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and volume_spike and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (bear_power_aligned[i] < 0 and bull_power_aligned[i] > 0 and volume_spike and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Weakening uptrend (Bull Power <= 0 OR Bear Power >= 0) OR price below 1d EMA34 (trend change)
            if (bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0 or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Weakening downtrend (Bear Power >= 0 OR Bull Power <= 0) OR price above 1d EMA34 (trend change)
            if (bear_power_aligned[i] >= 0 or bull_power_aligned[i] <= 0 or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals