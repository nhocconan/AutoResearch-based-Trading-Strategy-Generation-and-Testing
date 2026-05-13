#!/usr/bin/env python3
# 4h_TRIX_Zero_Cross_Volume_Spike_Trend_Filter
# Hypothesis: TRIX zero-line crossovers in the direction of 1d EMA50 trend, confirmed by volume spike.
# TRIX captures momentum reversals with low whipsaw. Zero-line cross provides clear entry/exit.
# Volume spike confirms institutional participation. EMA50 trend filter ensures alignment with higher timeframe momentum.
# Works in bull (long on zero-line cross up in uptrend) and bear (short on zero-line cross down in downtrend).
# Low frequency due to requirement of TRIX crossover + volume spike + trend alignment.

name = "4h_TRIX_Zero_Cross_Volume_Spike_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate TRIX on 4h data
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = (pd.Series(ema3).pct_change() * 100).values  # percentage
    
    # Volume spike: volume > 2.0 * 24-period average (2 days worth at 4h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(trix[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + daily uptrend + volume spike
            if trix[i-1] <= 0 and trix[i] > 0 and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + daily downtrend + volume spike
            elif trix[i-1] >= 0 and trix[i] < 0 and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR trend reversal
            if trix[i-1] >= 0 and trix[i] < 0:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR trend reversal
            if trix[i-1] <= 0 and trix[i] > 0:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals