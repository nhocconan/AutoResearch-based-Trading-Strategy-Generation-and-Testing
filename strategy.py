#!/usr/bin/env python3
"""
12h_KAMA_With_Trend_Filter_And_Volume_Confirmation
Hypothesis: Use KAMA trend direction on 12h timeframe filtered by weekly trend and volume spikes to reduce false signals.
KAMA adapts to market noise, making it effective in both trending and ranging conditions.
Long when KAMA trending up, weekly trend up, and volume spike; short when opposite.
Designed for 12h timeframe to capture medium-term moves with reduced whipsaw.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: weekly trend filter avoids counter-trend trades, volume filter reduces false signals.
"""

name = "12h_KAMA_With_Trend_Filter_And_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for KAMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close_prices, er_period=10, fast_ema=2, slow_ema=30):
        n = len(close_prices)
        kama_vals = np.full(n, np.nan)
        if n < er_period:
            return kama_vals
        
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close_prices, er_period))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        er = np.zeros(n)
        for i in range(er_period, n):
            if volatility[i] != 0:
                er[i] = change[i-er_period] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
        
        # Initialize KAMA
        kama_vals[er_period] = close_prices[er_period]
        for i in range(er_period+1, n):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close_prices[i] - kama_vals[i-1])
        
        return kama_vals
    
    # Calculate KAMA on 12h data
    kama_12h = kama(close_12h, 10, 2, 30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on 1w for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema20_1w = ema(close_1w, 20)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate volume filter (volume > 2.0x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA trending up, weekly uptrend, volume spike
            if (close[i] > kama_12h_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down, weekly downtrend, volume spike
            elif (close[i] < kama_12h_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down OR weekly trend turns down
            if (close[i] < kama_12h_aligned[i] or 
                close[i] < ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up OR weekly trend turns up
            if (close[i] > kama_12h_aligned[i] or 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals