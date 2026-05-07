#!/usr/bin/env python3
"""
6h_Aroon_Oscillator_12hTrend_Volume
Hypothesis: 6h Aroon Oscillator (25-period) detects early trend changes with 12h EMA50 trend filter and volume confirmation. Works in bull/bear markets by capturing momentum shifts early, avoiding whipsaws via trend alignment. Targets 15-35 trades/year to minimize fee drag on 6h timeframe.
"""
name = "6h_Aroon_Oscillator_12hTrend_Volume"
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
    
    # Get 12h data for Aroon calculation and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Aroon Oscillator on 12h high/low (25-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    period = 25
    
    def calculate_aroon(high_arr, low_arr, period):
        n_len = len(high_arr)
        aroon_up = np.full(n_len, np.nan)
        aroon_down = np.full(n_len, np.nan)
        for i in range(period-1, n_len):
            window_high = high_arr[i-period+1:i+1]
            window_low = low_arr[i-period+1:i+1]
            # Find periods since highest high and lowest low
            high_idx = np.argmax(window_high)
            low_idx = np.argmin(window_low)
            aroon_up[i] = ((period - 1 - high_idx) / (period - 1)) * 100
            aroon_down[i] = ((period - 1 - low_idx) / (period - 1)) * 100
        return aroon_up - aroon_down  # Aroon Oscillator
    
    aroon_osc = calculate_aroon(high_12h, low_12h, period)
    aroon_osc_aligned = align_htf_to_ltf(prices, df_12h, aroon_osc)
    
    # Calculate 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume filter: current 6h volume > 1.8 x 30-period average volume
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_avg * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 30)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(aroon_osc_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 6h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: Aroon Oscillator crosses above 0 with 12h uptrend and volume spike
            if (aroon_osc_aligned[i] > 0 and aroon_osc_aligned[i-1] <= 0 and 
                close[i] > ema_50_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: Aroon Oscillator crosses below 0 with 12h downtrend and volume spike
            elif (aroon_osc_aligned[i] < 0 and aroon_osc_aligned[i-1] >= 0 and 
                  close[i] < ema_50_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: Aroon Oscillator crosses back through 0 (momentum shift)
            if position == 1 and aroon_osc_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and aroon_osc_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals