#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume
# Hypothesis: Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and volume confirmation
# captures institutional pivot reversals with higher timeframe trend alignment.
# Works in bull/bear markets by filtering counter-trend breaks.
# Target: 20-30 trades/year to minimize fee drag while capturing strong moves.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r1 = close_1d + (high_1d - low_1d) * 1.12 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.12 / 12
    
    # Calculate daily volume average (20-period) for volume filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    vol_ma_20_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate volume spike on 4h timeframe
    vol_ma_20_4h_calc = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20_4h_calc)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with uptrend (above 12h EMA50) and volume
            if close[i] > r1_4h[i] and close[i] > ema_50_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with downtrend (below 12h EMA50) and volume
            elif close[i] < s1_4h[i] and close[i] < ema_50_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below 12h EMA50 (trend change)
            if close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above 12h EMA50 (trend change)
            if close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals