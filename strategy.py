#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeS
# Hypothesis: Uses weekly EMA50 trend and monthly volatility filter with daily Camarilla R1/S1 breakouts.
# Weekly trend filter (EMA50) reduces whipsaws; monthly volatility filter ensures trades occur in stable regimes.
# Designed for 12h timeframe to target 50-150 total trades over 4 years.
# Works in bull markets (price above weekly EMA50 + breaks R1 with volume) and bear markets (price below weekly EMA50 + breaks S1 with volume).

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeS"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R1, S1
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get monthly data for volatility filter (ATR ratio)
    df_1M = get_htf_data(prices, '1M')
    if len(df_1M) < 14:
        return np.zeros(n)
    
    high_1M = df_1M['high'].values
    low_1M = df_1M['low'].values
    close_1M = df_1M['close'].values
    tr_1M = np.maximum(high_1M - low_1M, np.absolute(high_1M - np.roll(close_1M, 1)), np.absolute(low_1M - np.roll(close_1M, 1)))
    tr_1M[0] = high_1M[0] - low_1M[0]
    atr_14_1M = pd.Series(tr_1M).rolling(window=14, min_periods=14).mean().values
    atr_50_1M = pd.Series(tr_1M).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_14_1M < atr_50_1M  # Low volatility regime
    
    # Align all indicators to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_1w_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    volatility_filter_12h = align_htf_to_ltf(prices, df_1M, volatility_filter, additional_delay_bars=0)
    
    # Volume spike filter on 12h (24-period average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_50_1w_12h[i]) or np.isnan(volatility_filter_12h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > R1, above weekly EMA50 trend, low volatility, volume spike
            if close[i] > r1_12h[i] and close[i] > ema_50_1w_12h[i] and volatility_filter_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < S1, below weekly EMA50 trend, low volatility, volume spike
            elif close[i] < s1_12h[i] and close[i] < ema_50_1w_12h[i] and volatility_filter_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < R1 or below weekly EMA50 trend
            if close[i] < r1_12h[i] or close[i] < ema_50_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > S1 or above weekly EMA50 trend
            if close[i] > s1_12h[i] or close[i] > ema_50_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals