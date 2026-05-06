#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray Bull/Bear Power with 6h EMA20 trend filter and volume spike confirmation
# Long when 1d Bull Power > 0 (bullish momentum) AND price > 6h EMA20 (uptrend) AND volume > 2.0 * avg_volume(20) on 6h
# Short when 1d Bear Power < 0 (bearish momentum) AND price < 6h EMA20 (downtrend) AND volume > 2.0 * avg_volume(20) on 6h
# Exit when Elder Power crosses zero (momentum shift) or price crosses EMA20
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures bull/bear power via EMA13, effective in both bull and bear markets
# 6h EMA20 trend filter ensures we trade with the intermediate-term trend
# Volume spike confirmation (2.0x) validates momentum while limiting overtrading
# Works in bull markets (buy strength) and bear markets (sell weakness)

name = "6h_ElderRay_BullBearPower_6hEMA20_Trend_VolumeSpike"
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
    if len(df_1d) < 13:  # Need at least 13 completed daily bars for EMA13
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray Bull Power and Bear Power
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema_13_1d   # Bear Power = Low - EMA13
    
    # Align 1d Elder Ray to 6h timeframe (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h EMA20
    ema_20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_20_6h[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum), price > 6h EMA20 (uptrend), volume spike, in session
            if (bull_power_aligned[i] > 0 and 
                close[i] > ema_20_6h[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish momentum), price < 6h EMA20 (downtrend), volume spike, in session
            elif (bear_power_aligned[i] < 0 and 
                  close[i] < ema_20_6h[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power crosses below 0 OR price crosses below 6h EMA20 (momentum/trend shift)
            if bull_power_aligned[i] <= 0 or close[i] <= ema_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power crosses above 0 OR price crosses above 6h EMA20 (momentum/trend shift)
            if bear_power_aligned[i] >= 0 or close[i] >= ema_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals