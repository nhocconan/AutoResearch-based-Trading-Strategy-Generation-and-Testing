#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_12hTrend_Volume
# Hypothesis: TRIX (triple smoothed EMA) zero crossovers combined with 12h trend filter and volume spikes provide reliable momentum signals.
# TRIX filters out market noise and identifies trend changes with less lag than MACD.
# Long when TRIX crosses above zero with 12h uptrend and volume confirmation.
# Short when TRIX crosses below zero with 12h downtrend and volume confirmation.
# Volume spike confirms institutional participation, reducing false signals.
# Works in bull markets (TRIX > 0 in uptrend) and bear markets (TRIX < 0 in downtrend).
# Target: 25-50 trades/year per symbol to minimize fee drag.

name = "4h_TRIX_ZeroCross_12hTrend_Volume"
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

    # Get 12h data for TRIX calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values

    # Calculate TRIX on 12h close: triple EMA of ROC
    # TRIX = EMA(EMA(EMA(ROC, 12), 12), 12) where ROC = (close/tmp - 1) * 100
    # Using period 12 as standard
    close_series = pd.Series(close_12h)
    roc = close_series.pct_change(periods=12) * 100
    ema1 = roc.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.values
    
    # 12h trend: EMA34
    ema34_12h = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume spike: volume > 2.0 * 4-period average (2 days worth at 4h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + 12h uptrend + volume spike
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and close[i] > ema34_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + 12h downtrend + volume spike
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and close[i] < ema34_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend reversal
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend reversal
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals