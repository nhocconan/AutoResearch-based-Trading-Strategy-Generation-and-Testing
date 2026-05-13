#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
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
    
    # Load 1D data ONCE for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    def calculate_camarilla(high, low, close):
        typical_price = (high + low + close) / 3
        range_hl = high - low
        
        # Camarilla levels
        R4 = close + range_hl * 1.1 / 2
        R3 = close + range_hl * 1.1 / 4
        R2 = close + range_hl * 1.1 / 6
        R1 = close + range_hl * 1.1 / 12
        
        S1 = close - range_hl * 1.1 / 12
        S2 = close - range_hl * 1.1 / 6
        S3 = close - range_hl * 1.1 / 4
        S4 = close - range_hl * 1.1 / 2
        
        return R1, R2, R3, R4, S1, S2, S3, S4
    
    # Calculate Camarilla levels for each day (using previous day's data)
    R1 = np.full_like(close_1d, np.nan)
    R2 = np.full_like(close_1d, np.nan)
    R3 = np.full_like(close_1d, np.nan)
    R4 = np.full_like(close_1d, np.nan)
    S1 = np.full_like(close_1d, np.nan)
    S2 = np.full_like(close_1d, np.nan)
    S3 = np.full_like(close_1d, np.nan)
    S4 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
        R1[i] = r1
        R2[i] = r2
        R3[i] = r3
        R4[i] = r4
        S1[i] = s1
        S2[i] = s2
        S3[i] = s3
        S4[i] = s4
    
    # Calculate 1D EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume spike (current volume > 2 * 20-period average)
    volume_series = pd.Series(volume)
    vol_avg_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_avg_20)
    
    # Align all 1D indicators to 6H timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after sufficient data for EMA34
        if (np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # LONG: Break above R3 with volume spike in uptrend
            if close[i] > R3_aligned[i] and volume_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike in downtrend
            elif close[i] < S3_aligned[i] and volume_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below R1 or trend reversal
            if close[i] < R1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above S1 or trend reversal
            if close[i] > S1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals