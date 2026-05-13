#!/usr/bin/env python3
# 12h_TRIX_ZeroCross_VolumeSpike_1dTrend
# Hypothesis: TRIX (triple smoothed EMA) zero-cross signals on 12h chart with volume spike confirmation and 1d trend filter (EMA34).
# Works in bull markets (TRIX crosses above zero in uptrend) and bear markets (TRIX crosses below zero in downtrend).
# Uses TRIX as momentum oscillator with low lag, volume confirmation to filter false signals, and trend filter to avoid counter-trend trades.
# Target: 15-30 trades/year to minimize fee drag on 12h timeframe.

name = "12h_TRIX_ZeroCross_VolumeSpike_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for TRIX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:  # Need enough for TRIX calculation
        return np.zeros(n)
    
    # Calculate TRIX on 12h: triple EMA of log(close)
    # TRIX = EMA(EMA(EMA(log(close)), 15), 15), 15) * 100
    log_close = np.log(df_12h['close'].values)
    
    # First EMA
    ema1 = pd.Series(log_close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second EMA
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Third EMA
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX calculation
    trix_raw = np.diff(ema3, prepend=ema3[0]) / ema3 * 100
    trix = pd.Series(trix_raw).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # Align TRIX to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: TRIX crosses above zero with volume confirmation in uptrend (close > EMA34)
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and volume_confirmed[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero with volume confirmation in downtrend (close < EMA34)
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and volume_confirmed[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend weakens (close < EMA34)
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend weakens (close > EMA34)
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals