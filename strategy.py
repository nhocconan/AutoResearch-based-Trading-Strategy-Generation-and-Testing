#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels (R1/S1) from prior 4h session for high-probability breakout entries
# 4h EMA50 confirms trend direction to avoid counter-trend trades in ranging markets
# Volume confirmation (>2.0x 20 EMA) ensures breakout has participation
# Discrete sizing 0.20 limits risk and reduces fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Works in both bull and bear: EMA50 ensures we only trade with the trend, Camarilla provides precise levels.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.

name = "1h_Camarilla_R1S1_4hEMA50_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter and Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Get 4h data for Camarilla pivot levels (R1, S1)
    if len(df_4h) < 1:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for prior 4h bar
    # Camarilla: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    camarilla_high = close_4h + 1.1 * (high_4h - low_4h) * 1.1 / 12
    camarilla_low = close_4h - 1.1 * (high_4h - low_4h) * 1.1 / 12
    
    # Shift by 1 to use prior completed 4h bar only (no look-ahead)
    camarilla_high_shifted = np.roll(camarilla_high, 1)
    camarilla_low_shifted = np.roll(camarilla_low, 1)
    camarilla_high_shifted[0] = np.nan  # first value invalid
    camarilla_low_shifted[0] = np.nan
    
    # Align Camarilla levels to 1h timeframe (completed 4h bar only)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high_shifted)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 + price above 4h EMA50 + volume spike
            if close[i] > camarilla_high_aligned[i] and close[i] > ema_50_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S1 + price below 4h EMA50 + volume spike
            elif close[i] < camarilla_low_aligned[i] and close[i] < ema_50_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot point (close) OR price crosses below 4h EMA50
            camarilla_pivot = (high_4h + low_4h + close_4h) / 3.0  # standard pivot point
            camarilla_pivot_shifted = np.roll(camarilla_pivot, 1)
            camarilla_pivot_shifted[0] = np.nan
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot_shifted)
            
            if not np.isnan(camarilla_pivot_aligned[i]) and (close[i] < camarilla_pivot_aligned[i] or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla pivot point (close) OR price crosses above 4h EMA50
            camarilla_pivot = (high_4h + low_4h + close_4h) / 3.0  # standard pivot point
            camarilla_pivot_shifted = np.roll(camarilla_pivot, 1)
            camarilla_pivot_shifted[0] = np.nan
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot_shifted)
            
            if not np.isnan(camarilla_pivot_aligned[i]) and (close[i] > camarilla_pivot_aligned[i] or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals