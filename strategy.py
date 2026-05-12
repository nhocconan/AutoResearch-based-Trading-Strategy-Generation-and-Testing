#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Use Camarilla pivot levels on 1h for mean-reversion entries, filtered by 4h EMA trend and volume spike.
# Long when price breaks above R1 with 4h uptrend and volume spike; short when breaks below S1 with 4h downtrend and volume spike.
# Exit on opposite Camarilla level (S1 for long, R1 for short) or trend failure.
# Designed for 15-30 trades/year to avoid fee drag. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for a single period.
    Returns R1, S1, R2, S2, R3, S3, R4, S4.
    """
    typical = (high + low + close) / 3.0
    range_val = high - low
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    R2 = close + range_val * 1.1 / 6
    S2 = close - range_val * 1.1 / 6
    R3 = close + range_val * 1.1 / 4
    S3 = close - range_val * 1.1 / 4
    R4 = close + range_val * 1.1 / 2
    S4 = close - range_val * 1.1 / 2
    return R1, S1, R2, S2, R3, S3, R4, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels on 4h data
    R1_4h, S1_4h, R2_4h, S2_4h, R3_4h, S3_4h, R4_4h, S4_4h = calculate_camarilla(high_4h, low_4h, close_4h)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average on 1h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h data to 1h timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure EMA is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA34 direction
        trend_up = close[i] > ema_34_4h_aligned[i]
        trend_down = close[i] < ema_34_4h_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: price breaks above R1 with 4h uptrend and volume spike
            if close[i] > R1_4h_aligned[i] and trend_up and vol_ok:
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S1 with 4h downtrend and volume spike
            elif close[i] < S1_4h_aligned[i] and trend_down and vol_ok:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: price breaks below S1 or trend fails
            if close[i] < S1_4h_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or trend fails
            if close[i] > R1_4h_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals