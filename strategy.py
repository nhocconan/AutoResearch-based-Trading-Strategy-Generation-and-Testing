#!/usr/bin/env python3
"""
12h_CCI_Trend_Filter_With_Volume
Hypothesis: Trade long when CCI(20) crosses above -100 with volume confirmation and price above 12h EMA50 (uptrend); short when CCI crosses below +100 with volume confirmation and price below EMA50 (downtrend). Uses volume spike (>1.5x 20-period avg) to confirm momentum. Designed for low trade frequency (~12-25/year) to minimize fee drag and work in both bull and bear markets via trend filter.
"""

name = "12h_CCI_Trend_Filter_With_Volume"
timeframe = "12h"
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
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate CCI(20)
    def cci(high, low, close, period):
        tp = (high + low + close) / 3.0
        cci_values = np.full_like(tp, np.nan)
        if len(tp) < period:
            return cci_values
        ma = np.full_like(tp, np.nan)
        md = np.full_like(tp, np.nan)
        for i in range(period-1, len(tp)):
            ma[i] = np.mean(tp[i-period+1:i+1])
            md[i] = np.mean(np.abs(tp[i-period+1:i+1] - ma[i]))
            if md[i] != 0:
                cci_values[i] = (tp[i] - ma[i]) / (0.015 * md[i])
        return cci_values
    
    cci_values = cci(high, low, close, 20)
    
    # Calculate volume spike (>1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cci_values[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CCI crosses above -100 with volume spike and price above EMA50
            if cci_values[i] > -100 and cci_values[i-1] <= -100 and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: CCI crosses below +100 with volume spike and price below EMA50
            elif cci_values[i] < 100 and cci_values[i-1] >= 100 and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI crosses below +100 or trend turns down
            if cci_values[i] < 100 and cci_values[i-1] >= 100 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI crosses above -100 or trend turns up
            if cci_values[i] > -100 and cci_values[i-1] <= -100 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals