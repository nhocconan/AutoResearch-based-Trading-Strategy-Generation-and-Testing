#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R regime
# Long when 6h Bull Power > 0 AND 1d Williams %R < -80 (oversold) AND volume > 1.5 * avg_volume(20)
# Short when 6h Bear Power < 0 AND 1d Williams %R > -20 (overbought) AND volume > 1.5 * avg_volume(20)
# Exit when Elder Power crosses zero or Williams %R reverts to neutral (-50)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures bull/bear power via EMA13, effective in both trending and ranging markets
# Williams %R on 1d provides overbought/oversold signals from higher timeframe
# Volume confirmation filters weak signals
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)

name = "6h_ElderRay_WilliamsR_Volume"
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
    
    # Get 6h data ONCE before loop for Elder Ray (EMA13)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:  # Need sufficient data for EMA13
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_series_6h = pd.Series(close_6h)
    ema_13_6h = close_series_6h.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Bull Power and Bear Power
    bull_power_6h = high_6h - ema_13_6h  # Bull Power = High - EMA13
    bear_power_6h = low_6h - ema_13_6h   # Bear Power = Low - EMA13
    
    # Align 6h Elder Ray values to 6h timeframe (wait for completed 6h bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power_6h)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power_6h)
    
    # Get 1d data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need sufficient data for Williams %R(14)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    close_series_1d = pd.Series(close_1d)
    highest_high_14 = high_series_1d.rolling(window=14, min_periods=14).max().values
    lowest_low_14 = low_series_1d.rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_14 - close_series_1d) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r_1d = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_1d)
    
    # Align 1d Williams %R to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Williams %R < -80 (oversold) AND volume confirmation
            if (bull_power_aligned[i] > 0 and williams_r_aligned[i] < -80 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Williams %R > -20 (overbought) AND volume confirmation
            elif (bear_power_aligned[i] < 0 and williams_r_aligned[i] > -20 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power crosses below zero OR Williams %R reverts above -50
            if bull_power_aligned[i] <= 0 or williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power crosses above zero OR Williams %R reverts below -50
            if bear_power_aligned[i] >= 0 or williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals