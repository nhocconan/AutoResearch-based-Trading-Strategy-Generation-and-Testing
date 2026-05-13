#!/usr/bin/env python3
# Hypothesis: 1d Williams Alligator trend filter with 1h KAMA entry timing and volume confirmation.
# Uses 1w EMA13/8/5 for Alligator jaws/teeth/lips (HTF) to define trend direction.
# Enters on 1h KAMA cross above/below price with volume > 1.5x 20-bar average.
# Exits on Alligator reversal (jaws cross below/above lips) or opposite signal.
# Designed for low frequency (target 30-100 trades over 4 years) to minimize fee drag.
# Works in bull/bear by following Alligator trend and requiring volume confirmation.

name = "1d_WilliamsAlligator_1hKAMA_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w Williams Alligator (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Alligator: Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    def smma(arr, period):
        # Smoothed Moving Average: first value = SMA, then recursive
        sma = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(arr, np.nan, dtype=float)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(arr)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + arr[i]) / period
        return smma_vals
    
    jaw = smma(close_1w, 13)
    teeth = smma(close_1w, 8)
    lips = smma(close_1w, 5)
    
    # Align Alligator components to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate 1h KAMA for entry timing (MTF)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    close_1h = df_1h['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1h, n=10))
    volatility = np.sum(np.abs(np.diff(close_1h, n=1)), axis=0)
    er = np.divide(change, volatility, out=np.full_like(change, np.nan), where=volatility!=0)
    # Smoothing constants
    fastest = 2.0 / (2 + 1)
    slowest = 2.0 / (30 + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # KAMA calculation
    kama = np.full_like(close_1h, np.nan, dtype=float)
    kama[9] = close_1h[9]  # start after 10 periods
    for i in range(10, len(close_1h)):
        if np.isnan(sc[i-10]) or np.isnan(kama[i-1]):
            kama[i] = close_1h[i]
        else:
            kama[i] = kama[i-1] + sc[i-10] * (close_1h[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1h, kama)
    
    # Volume confirmation: 20-bar average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(kama_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator bullish (jaws > teeth > lips) AND price > KAMA AND volume spike
            if (jaw_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > lips_aligned[i] and
                close[i] > kama_aligned[i] and
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Alligator bearish (jaws < teeth < lips) AND price < KAMA AND volume spike
            elif (jaw_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < lips_aligned[i] and
                  close[i] < kama_aligned[i] and
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator turns bearish (jaws < lips) OR opposite signal
            if jaw_aligned[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator turns bullish (jaws > lips) OR opposite signal
            if jaw_aligned[i] > lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals