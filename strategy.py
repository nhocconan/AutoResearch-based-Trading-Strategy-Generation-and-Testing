#!/usr/bin/env python3
# 1D_CAMARILLA_R3_S3_BREAKOUT_WEEKLYTREND_VOLUME_CONFIRMATION
# Hypothesis: Camarilla R3/S3 levels from daily pivots provide strong reversal signals when aligned with weekly trend (EMA34) and volume spikes.
# In uptrend, buy near S3 support; in downtrend, sell near R3 resistance.
# Volume confirmation filters false breakouts. Works in bull/bear markets by trading with the higher timeframe trend.

name = "1D_CAMARILLA_R3_S3_BREAKOUT_WEEKLYTREND_VOLUME_CONFIRMATION"
timeframe = "1d"
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
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly data for trend filter and volume average
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    camarilla_r3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_s3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    
    # Weekly EMA for trend filter (34-period)
    ema34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly volume average (20-period) for spike detection
    vol_ma = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x weekly average
        volume_spike = volume[i] > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # LONG: Price at S3 support in uptrend with volume spike
            if (close[i] <= s3_aligned[i] * 1.005 and  # Allow small buffer
                close[i] > ema34_aligned[i] and        # Above weekly trend
                volume_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R3 resistance in downtrend with volume spike
            elif (close[i] >= r3_aligned[i] * 0.995 and  # Allow small buffer
                  close[i] < ema34_aligned[i] and        # Below weekly trend
                  volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R3 or weekly trend breaks down
            if (close[i] >= r3_aligned[i] * 0.995 or  # Near R3
                close[i] < ema34_aligned[i]):         # Below weekly trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S3 or weekly trend breaks up
            if (close[i] <= s3_aligned[i] * 1.005 or  # Near S3
                close[i] > ema34_aligned[i]):         # Above weekly trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals