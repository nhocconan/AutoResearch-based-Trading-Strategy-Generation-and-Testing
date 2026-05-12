# 6H_EAGLE_CLAW_EMA12_CROSS_4H_WAVE_TREND
# Hypothesis: Uses EMA12 crossover on 6h timeframe for momentum, filtered by 4H WaveTrend oscillator (overbought/oversold).
# WaveTrend identifies exhaustion points in trends, allowing entries in direction of higher timeframe trend.
# Works in bull markets: buy dips in uptrend when WT oversold. Works in bear markets: sell rallies in downtrend when WT overbought.
# Combines fast entry (EMA cross) with WT exhaustion filter to avoid chasing extended moves.
# Target: 60-120 total trades over 4 years (15-30/year).

name = "6H_EAGLE_CLAW_EMA12_CROSS_4H_WAVE_TREND"
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
    
    # 4H data for WaveTrend indicator
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # WaveTrend Oscillator calculation
    # Typical price and EMA smoothing
    ap = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    esa = pd.Series(ap).ewm(span=10, adjust=False, min_periods=10).mean()
    d = pd.Series(abs(ap - esa)).ewm(span=10, adjust=False, min_periods=10).mean()
    ci = (ap - esa) / (0.015 * d)
    tci = pd.Series(ci).ewm(span=21, adjust=False, min_periods=21).mean()
    wt1 = tci.values
    wt2 = pd.Series(wt1).ewm(span=4, adjust=False, min_periods=4).mean()
    
    # WaveTrend levels: overbought > 60, oversold < -60
    wt1_aligned = align_htf_to_ltf(prices, df_4h, wt1)
    wt2_aligned = align_htf_to_ltf(prices, df_4h, wt2)
    
    # 6H EMA12 for entry signal
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 12)  # Ensure EMA and WT are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(wt1_aligned[i]) or np.isnan(wt2_aligned[i]) or np.isnan(ema12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: EMA12 crosses above previous value AND WaveTrend oversold (< -60)
            if (ema12[i] > ema12[i-1] and wt1_aligned[i] < -60):
                signals[i] = 0.25
                position = 1
            # SHORT: EMA12 crosses below previous value AND WaveTrend overbought (> 60)
            elif (ema12[i] < ema12[i-1] and wt1_aligned[i] > 60):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: EMA12 turns down OR WaveTrend overbought (> 60)
            if (ema12[i] < ema12[i-1]) or (wt1_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: EMA12 turns up OR WaveTrend oversold (< -60)
            if (ema12[i] > ema12[i-1]) or (wt1_aligned[i] < -60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals