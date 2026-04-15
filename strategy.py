#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h Trend Filter + Volume Confirmation
# Elder Ray measures bull/bear power relative to EMA13. Long when bull power > 0 and rising, short when bear power < 0 and falling.
# Uses 12h EMA50 as trend filter to avoid counter-trend trades. Volume confirmation reduces false signals.
# Works in bull markets (buy strength) and bear markets (sell weakness). Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Smooth the power signals (3-period)
    bull_power_smooth = pd.Series(bull_power).rolling(window=3, min_periods=3).mean().values
    bear_power_smooth = pd.Series(bear_power).rolling(window=3, min_periods=3).mean().values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or
            np.isnan(ema50_12h_aligned[i])):
            continue
        
        # Long entry: bull power positive AND rising, price above 12h EMA50, volume confirmation
        if (bull_power_smooth[i] > 0 and
            bull_power_smooth[i] > bull_power_smooth[i-1] and
            close[i] > ema50_12h_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bear power negative AND falling, price below 12h EMA50, volume confirmation
        elif (bear_power_smooth[i] < 0 and
              bear_power_smooth[i] < bear_power_smooth[i-1] and
              close[i] < ema50_12h_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite signal appears or power crosses zero
        elif position == 1 and (bear_power_smooth[i] < 0 or bull_power_smooth[i] < 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bull_power_smooth[i] > 0 or bear_power_smooth[i] > 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0