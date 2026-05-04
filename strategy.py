#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) from prior 1d session for high-probability breakout entries
# 1d EMA34 confirms trend direction to avoid counter-trend trades in ranging markets
# Volume confirmation (>2.0x 20 EMA) ensures breakout has participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h.
# Works in both bull and bear: EMA34 ensures we only trade with the trend, Camarilla provides precise levels.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeConfirm_Session"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 1d data for Camarilla pivot levels (R3, S3)
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for prior 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    camarilla_high = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_low = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    
    # Shift by 1 to use prior completed 1d bar only (no look-ahead)
    camarilla_high_shifted = np.roll(camarilla_high, 1)
    camarilla_low_shifted = np.roll(camarilla_low, 1)
    camarilla_high_shifted[0] = np.nan  # first value invalid
    camarilla_low_shifted[0] = np.nan
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high_shifted)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + price above 1d EMA34 + volume spike
            if close[i] > camarilla_high_aligned[i] and close[i] > ema_34_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + price below 1d EMA34 + volume spike
            elif close[i] < camarilla_low_aligned[i] and close[i] < ema_34_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot point (close) OR price crosses below 1d EMA34
            camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0  # standard pivot point
            camarilla_pivot_shifted = np.roll(camarilla_pivot, 1)
            camarilla_pivot_shifted[0] = np.nan
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_shifted)
            
            if not np.isnan(camarilla_pivot_aligned[i]) and (close[i] < camarilla_pivot_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot point (close) OR price crosses above 1d EMA34
            camarilla_pivot = (high_1d + low_1d + close_1d) / 3.0  # standard pivot point
            camarilla_pivot_shifted = np.roll(camarilla_pivot, 1)
            camarilla_pivot_shifted[0] = np.nan
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_shifted)
            
            if not np.isnan(camarilla_pivot_aligned[i]) and (close[i] > camarilla_pivot_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals