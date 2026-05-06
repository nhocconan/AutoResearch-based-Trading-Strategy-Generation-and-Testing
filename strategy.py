#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot R4/S4 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1w Camarilla R4 level AND 1w EMA34 is rising AND volume > 2.0 * avg_volume(20) on 1d
# Short when price breaks below 1w Camarilla S4 level AND 1w EMA34 is falling AND volume > 2.0 * avg_volume(20) on 1d
# Exit when price crosses the 1w Camarilla pivot point (midpoint of R4/S4)
# Uses discrete sizing 0.30 to balance profit potential and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Camarilla R4/S4 provides strong breakout levels with fewer false signals than daily levels
# 1w EMA34 ensures we trade with the weekly trend while reducing noise
# High volume threshold (2.0x) controls trade frequency while capturing genuine institutional breakouts
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by trading with the 1w trend

name = "1d_1wCamarilla_R4S4_Breakout_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla pivot and EMA34 calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 completed weekly bars for EMA34
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels (R4, S4, and pivot point)
    # Camarilla formulas:
    # Pivot = (H + L + C) / 3
    # R4 = C + (H - L) * 1.1 / 2
    # S4 = C - (H - L) * 1.1 / 2
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r4_1w = close_1w + (high_1w - low_1w) * 1.1 / 2.0
    s4_1w = close_1w - (high_1w - low_1w) * 1.1 / 2.0
    
    # Align 1w Camarilla levels to 1d timeframe (wait for completed 1w bar)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 1w EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R4, EMA34 rising, volume spike
            if (close[i] > r4_aligned[i] and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1w Camarilla S4, EMA34 falling, volume spike
            elif (close[i] < s4_aligned[i] and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below the 1w Camarilla pivot point
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above the 1w Camarilla pivot point
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals