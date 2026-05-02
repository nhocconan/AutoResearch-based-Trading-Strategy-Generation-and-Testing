#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA50 trend + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In strong trends, one power dominates while the other weakens. Enter long when Bull Power rising AND Bear Power negative (weak bears)
# Enter short when Bear Power falling AND Bull Power positive (weak bulls). Uses 1d EMA50 for higher-timeframe trend filter.
# Volume spike confirms institutional participation. Works in both bull and bear markets by aligning with daily trend.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.

name = "6h_ElderRay_1dEMA50_Volume"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    if len(close) < 13:
        return np.zeros(n)
    
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Smooth the power signals to reduce noise (2-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    # Volume confirmation: 1.8x 20-period average
    if len(volume) < 20:
        return np.zeros(n)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 13, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power rising (positive AND increasing) AND Bear Power negative (weak bears) 
            # AND price above 1d EMA50 (bullish HTF trend) AND volume spike
            bull_rising = bull_power_smooth[i] > bull_power_smooth[i-1]
            if (bull_rising and 
                bull_power_smooth[i] > 0 and 
                bear_power_smooth[i] < 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power falling (negative AND decreasing) AND Bull Power positive (weak bulls)
            # AND price below 1d EMA50 (bearish HTF trend) AND volume spike
            elif (bear_power_smooth[i] < bear_power_smooth[i-1] and 
                  bear_power_smooth[i] < 0 and 
                  bull_power_smooth[i] > 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear Power becomes positive (bulls weakening) OR price breaks below 1d EMA50 (trend change)
            if bear_power_smooth[i] > 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power becomes negative (bears weakening) OR price breaks above 1d EMA50 (trend change)
            if bull_power_smooth[i] < 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals