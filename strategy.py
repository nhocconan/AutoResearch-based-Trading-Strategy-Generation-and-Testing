#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_4hTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) act as support/resistance; breakouts with
# 4h trend and volume filter capture strong moves. Uses 4h EMA50 for trend to avoid lag from higher timeframes,
# enabling faster adaptation to trend changes while maintaining fewer trades via volume confirmation.
# Works in bull (breakouts) and bear (mean reversion at extremes) with tight entries to avoid overtrading.

name = "4h_Camarilla_R1_S1_4hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA50 trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    
    # Align 4h trend to 4h (no delay needed as it's same timeframe)
    trend_4h_up_aligned = trend_4h_up.astype(float)
    trend_4h_down_aligned = trend_4h_down.astype(float)
    
    # Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    R1 = close_1d + 1.1 * range_1d / 12
    S1 = close_1d - 1.1 * range_1d / 12
    
    # Align Camarilla levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        if position == 0:
            # Long: break above R1 with 4h uptrend and volume spike
            if (close[i] > R1_aligned[i] and 
                trend_4h_up_aligned[i] > 0.5 and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with 4h downtrend and volume spike
            elif (close[i] < S1_aligned[i] and 
                  trend_4h_down_aligned[i] > 0.5 and volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below S1 or trend fails
            if (close[i] < S1_aligned[i] or 
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above R1 or trend fails
            if (close[i] > R1_aligned[i] or 
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals