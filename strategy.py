#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. Goes long when Bull Power > 0 and Bear Power < 0
# with volume spike and 12h EMA34 uptrend. Goes short when Bear Power < 0 and Bull Power < 0
# with volume spike and 12h EMA34 downtrend. Designed for 12-37 trades/year to minimize fee drag.
# Works in bull markets via strong bull power and in bear markets via strong bear power.

name = "6h_ElderRay_Index_12hEMA34_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Elder Ray and EMA34 - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA13 for Elder Ray on 12h data
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_12h = high_12h - ema13_12h  # Bull Power = High - EMA13
    bear_power_12h = low_12h - ema13_12h   # Bear Power = Low - EMA13
    
    # Align Elder Ray to 6h timeframe (wait for completed 12h bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # Get 12h data for EMA34 trend filter - ONCE before loop
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 6h timeframe (wait for completed 12h bar)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND volume spike AND 12h EMA34 uptrend
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                volume_spike[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power < 0 AND volume spike AND 12h EMA34 downtrend
            elif (bear_power_aligned[i] < 0 and 
                  bull_power_aligned[i] < 0 and 
                  volume_spike[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power >= 0 (bull power fading) OR trend reverses
            if bear_power_aligned[i] >= 0 or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power >= 0 (bear power fading) OR trend reverses
            if bull_power_aligned[i] >= 0 or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals