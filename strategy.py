#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullPower_1dTrend_Volume"
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
    
    # Get daily data for trend and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA13 for trend filter
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Daily EMA13 for Elder Ray calculation
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align Elder Ray components
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power positive (buying pressure) + price above daily EMA13 + volume confirmation
            if (bull_power_1d_aligned[i] > 0 and 
                close[i] > ema_13_1d_aligned[i] and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: Bear power negative (selling pressure) + price below daily EMA13 + volume confirmation
            elif (bear_power_1d_aligned[i] < 0 and 
                  close[i] < ema_13_1d_aligned[i] and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull power turns negative OR price below daily EMA13
            if bull_power_1d_aligned[i] <= 0 or close[i] < ema_13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear power turns positive OR price above daily EMA13
            if bear_power_1d_aligned[i] >= 0 or close[i] > ema_13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Calculate high and low arrays for Elder Ray (moved inside loop for efficiency)
high_1d = df_1d['high'].values
low_1d = df_1d['low'].values
bull_power_1d = high_1d - ema_13_1d
bear_power_1d = low_1d - ema_13_1d

# Re-align after calculation
bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)