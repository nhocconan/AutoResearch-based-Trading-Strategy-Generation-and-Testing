#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_12hTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from daily data act as strong support/resistance.
Breakouts with volume confirmation and 12h trend alignment capture institutional moves.
4h timeframe balances trade frequency and responsiveness. Works in bull markets via long
breakouts and bear markets via short breakdowns. Uses volume spike (>2x 20-day average)
and 12h EMA trend filter to avoid false breakouts.
"""

name = "4h_Camarilla_R1_S1_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # R2 = Close + (High - Low) * 1.1/6
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # S2 = Close - (High - Low) * 1.1/6
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    diff = high_1d - low_1d
    camarilla_r1 = close_1d + diff * 1.1 / 12.0
    camarilla_s1 = close_1d - diff * 1.1 / 12.0
    
    # Align Camarilla levels to 4h (based on prior day's data)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Get 12h data for volume average
    volume_12h = df_12h['volume'].values
    # 20-period average volume on 12h
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    # 4h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34) and volume average (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(vol_avg_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 12h trend filter
        uptrend_12h = close[i] > ema34_12h_aligned[i]
        downtrend_12h = close[i] < ema34_12h_aligned[i]
        
        # Volume filter: current 4h volume > 2.0x average 12h volume
        volume_spike = volume[i] > vol_avg_12h_aligned[i] * 2.0
        
        if position == 0:
            # Long entry: price breaks above R1 + uptrend + volume spike
            if high[i] > camarilla_r1_aligned[i] and uptrend_12h and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + downtrend + volume spike
            elif low[i] < camarilla_s1_aligned[i] and downtrend_12h and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S1 or trend fails
            if low[i] < camarilla_s1_aligned[i] or not uptrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R1 or trend fails
            if high[i] > camarilla_r1_aligned[i] or not downtrend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals