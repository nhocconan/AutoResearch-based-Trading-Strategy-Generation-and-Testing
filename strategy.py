#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 6h EMA20 trend filter and volume confirmation
# Long when 1d Bull Power > 0 (price > EMA13) AND 6h EMA20 rising AND volume > 1.5 * avg_volume(20) on 6h
# Short when 1d Bear Power < 0 (price < EMA13) AND 6h EMA20 falling AND volume > 1.5 * avg_volume(20) on 6h
# Exit when Elder Power crosses zero (mean reversion to fair value)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures bull/bear power relative to EMA13, providing trend strength confirmation
# 6h EMA20 trend filter ensures we trade with the intermediate-term trend
# Volume spike confirmation (1.5x) validates breakout strength while limiting overtrading
# Works in bull markets (buy strength) and bear markets (sell weakness)

name = "6h_ElderRay_BullBearPower_6hEMA20_Trend_VolumeConfirm"
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
    
    # Get 1d data ONCE before loop for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed daily bars for EMA13
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align 1d Elder Ray to 6h timeframe (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h EMA20 for trend filter
    ema_20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_20_6h[i]) or np.isnan(ema_20_6h[i-1]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (price > EMA13), 6h EMA20 rising, volume spike, in session
            if (bull_power_aligned[i] > 0 and 
                ema_20_6h[i] > ema_20_6h[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (price < EMA13), 6h EMA20 falling, volume spike, in session
            elif (bear_power_aligned[i] < 0 and 
                  ema_20_6h[i] < ema_20_6h[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power crosses below zero (loss of bullish strength)
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power crosses above zero (loss of bearish strength)
            if bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals