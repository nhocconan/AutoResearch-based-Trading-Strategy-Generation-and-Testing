#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 in uptrend (price > EMA34) with volume spike.
# Short when price breaks below S3 in downtrend (price < EMA34) with volume spike.
# Exit when price returns to the Camarilla pivot (midpoint between S3 and R3).
# Uses proper risk control via position sizing (0.25) and avoids overtrading via strict conditions.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Daily data for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Camarilla R3, S3 and pivot (midpoint)
    r3 = pc + 1.1 * (ph - pl) / 6
    s3 = pc - 1.1 * (ph - pl) / 6
    pivot = (r3 + s3) / 2  # midpoint between R3 and S3
    
    # Calculate 1-day EMA34 for trend filter
    close_1d = pc
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after sufficient data for EMA34
        if np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(pivot_4h[i]) or np.isnan(ema34_4h[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 in uptrend with volume confirmation
            if close[i] > r3_4h[i] and close[i] > ema34_4h[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 in downtrend with volume confirmation
            elif close[i] < s3_4h[i] and close[i] < ema34_4h[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot level
            if close[i] <= pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot level
            if close[i] >= pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals