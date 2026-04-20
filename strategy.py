#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_Volume_4hTrend_Filter
Hypothesis: Trade Camarilla R1/S1 breakouts on 1h with volume confirmation, filtered by 4h trend direction (EMA20).
Long when price breaks above R1 with volume spike and 4h uptrend; short when breaks below S1 with volume spike and 4h downtrend.
Uses volume spike (volume > 2x 20-period average) to confirm breakout strength.
Target: 60-150 total trades over 4 years (15-37/year) with position size 0.20 to control risk and avoid overtrading.
Works in bull/bear: 4h trend filter avoids counter-trend trades, volume confirmation reduces false breakouts.
"""
name = "1h_Camarilla_R1S1_Breakout_Volume_4hTrend_Filter"
timeframe = "1h"
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
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema20_4h = ema(close_4h, 20)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate volume spike (volume > 2x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma20)
    
    # Calculate Camarilla levels from previous hour (using 1h data)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    # Previous hour's range
    range_prev = high_shift - low_shift
    
    # Camarilla levels (using previous hour's close as base)
    R1 = close_shift + 1.1 * range_prev / 12
    S1 = close_shift - 1.1 * range_prev / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND 4h uptrend (price > EMA20)
            if close[i] > R1[i] and volume_spike[i] and close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume spike AND 4h downtrend (price < EMA20)
            elif close[i] < S1[i] and volume_spike[i] and close[i] < ema20_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR 4h trend turns down
            if close[i] < S1[i] or close[i] < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above R1 OR 4h trend turns up
            if close[i] > R1[i] or close[i] > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals