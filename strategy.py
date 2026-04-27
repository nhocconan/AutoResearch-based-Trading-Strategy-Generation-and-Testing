#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-day EMA50 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and Bear Power < 0 (bullish momentum), price > daily EMA50, volume > 1.5x average.
# Short when Bear Power < 0 and Bull Power < 0 (bearish momentum), price < daily EMA50, volume > 1.5x average.
# Exit when Elder Ray signals weaken (Bull Power <= 0 for long, Bear Power >= 0 for short).
# Uses daily EMA50 for trend alignment and Elder Ray for momentum confirmation.
# Target: 50-150 total trades over 4 years to minimize fee drag while capturing momentum shifts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray (using daily data)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power on daily timeframe
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average (moderate to balance signal frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: Bull Power > 0 (strong buying pressure), Bear Power < 0 (weak selling),
        # price above daily EMA50 (uptrend), volume confirmation
        if (bull_power_aligned[i] > 0 and 
            bear_power_aligned[i] < 0 and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: Bear Power < 0 (strong selling pressure), Bull Power < 0 (weak buying),
        # price below daily EMA50 (downtrend), volume confirmation
        elif (bear_power_aligned[i] < 0 and 
              bull_power_aligned[i] < 0 and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit long when Bull Power weakens (<= 0) - momentum fading
        elif position == 1 and bull_power_aligned[i] <= 0:
            signals[i] = 0.0
            position = 0
        # Exit short when Bear Power weakens (>= 0) - selling pressure fading
        elif position == -1 and bear_power_aligned[i] >= 0:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1dEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0