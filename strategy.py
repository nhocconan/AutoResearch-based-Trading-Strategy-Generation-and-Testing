#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels with 4h EMA50 trend filter and volume confirmation
# Long when price touches/bounces off Camarilla S1 level AND 4h EMA50 is rising AND 4h volume > 1.2 * avg_volume(20)
# Short when price touches/bounces off Camarilla R1 level AND 4h EMA50 is falling AND 4h volume > 1.2 * avg_volume(20)
# Exit when price reaches Camarilla midpoint (R1/S1 average) or opposite Camarilla level
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Uses discrete sizing 0.25 to balance profitability and fee drag
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# 1d Camarilla provides robust support/resistance levels from higher timeframe
# 4h EMA50 ensures we trade with the intermediate trend while reducing whipsaws
# Volume confirmation filters out low-conviction bounces
# Works in bull markets (buying dips at S1) and bear markets (selling rallies at R1)

name = "4h_1dCamarilla_S1R1_Bounce_4hEMA50_Trend_Volume_Session"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:  # Need at least 1 completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (S1, R1, midpoint)
    # Camarilla formulas: 
    # S1 = close - (high - low) * 1.1 / 12
    # R1 = close + (high - low) * 1.1 / 12
    # Mid = (S1 + R1) / 2
    daily_range = high_1d - low_1d
    camarilla_s1 = close_1d - (daily_range * 1.1 / 12)
    camarilla_r1 = close_1d + (daily_range * 1.1 / 12)
    camarilla_mid = (camarilla_s1 + camarilla_r1) / 2
    
    # Align 1d Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Get 4h data ONCE before loop for EMA50 trend filter and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need at least 50 completed 4h bars for EMA50
        return np.zeros(n)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume confirmation: volume > 1.2 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_4h > (1.2 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches/bounces off Camarilla S1, EMA50 rising, volume spike
            # Check for bounce: price >= S1 and was below S1 in previous bar
            if (close[i] >= camarilla_s1_aligned[i] and close[i-1] < camarilla_s1_aligned[i-1] and 
                ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches/bounces off Camarilla R1, EMA50 falling, volume spike
            # Check for bounce: price <= R1 and was above R1 in previous bar
            elif (close[i] <= camarilla_r1_aligned[i] and close[i-1] > camarilla_r1_aligned[i-1] and 
                  ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches Camarilla midpoint or R1 (take profit)
            if close[i] >= camarilla_mid_aligned[i] or close[i] >= camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches Camarilla midpoint or S1 (take profit)
            if close[i] <= camarilla_mid_aligned[i] or close[i] <= camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals