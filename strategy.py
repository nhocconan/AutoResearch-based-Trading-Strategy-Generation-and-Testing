#!/usr/bin/env python3
# 6h_ElderRay_Power_Breakout_1dTrend_Volume
# Hypothesis: Elder Ray (Bull/Bear Power) + 1d trend filter + volume spikes on 6h timeframe.
# Uses 13-period EMA for power calculation, with entries when power crosses zero with volume confirmation.
# Works in both bull/bear by adapting to trend direction via 1d EMA50 filter.
# Target: 15-30 trades/year to minimize fee drag while capturing strong momentum moves.

name = "6h_ElderRay_Power_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray (same for both timeframes)
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Get 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)  # Wait for 1d close
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)  # Wait for 1d close
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)   # Wait for 1d close
    
    # Volume spike filter on 6h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema_50_1d_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Bull Power crosses above zero, above 1d EMA50 trend, volume spike
            if (bull_power_6h[i] > 0 and bull_power_6h[i-1] <= 0 and 
                close[i] > ema_50_1d_6h[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Bear Power crosses below zero, below 1d EMA50 trend, volume spike
            elif (bear_power_6h[i] < 0 and bear_power_6h[i-1] >= 0 and 
                  close[i] < ema_50_1d_6h[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: Bull Power crosses below zero OR trend fails
            if bull_power_6h[i] < 0 or close[i] < ema_50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power crosses above zero OR trend fails
            if bear_power_6h[i] > 0 or close[i] > ema_50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals