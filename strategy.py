#!/usr/bin/env python3
name = "4h_TRIX_Zero_Cross_Volume_Confirm_Trend_Filter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # TRIX on 1d: TRIX = EMA(EMA(EMA(close, 15), 15), 15) then percent change
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix_raw.values
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX signal line to 4h (needs extra delay for signal confirmation)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal, additional_delay_bars=1)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: > 2.0x 24-period average (6 trading days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Wait for TRIX and EMA50
    
    for i in range(start_idx, n):
        if np.isnan(trix_signal_aligned[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike in uptrend
            if trix_signal_aligned[i] > 0 and trix_signal_aligned[i-1] <= 0 and vol_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike in downtrend
            elif trix_signal_aligned[i] < 0 and trix_signal_aligned[i-1] >= 0 and vol_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or trend turns down
            if trix_signal_aligned[i] < 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or trend turns up
            if trix_signal_aligned[i] > 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX (triple exponential moving average) zero-cross with volume confirmation and 1d EMA50 trend filter.
# TRIX is a momentum oscillator that filters out insignificant cycles. Zero-cross signals momentum shifts.
# Long when TRIX crosses above zero with volume spike in uptrend (price > EMA50).
# Short when TRIX crosses below zero with volume spike in downtrend (price < EMA50).
# Volume spike (>2.0x 24-period average) ensures conviction behind the momentum shift.
# Works in both bull and bear markets by capturing momentum shifts in either direction.
# Discrete position size (0.25) minimizes churn. Target ~25-40 trades/year (~100-160 total over 4 years).