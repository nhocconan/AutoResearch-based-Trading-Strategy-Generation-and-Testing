# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "4h_TRIX_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _ema(arr, span):
    """Exponential moving average with proper min_periods."""
    return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().values

def _trix(close, period):
    """TRIX: Triple EMA rate of change."""
    e1 = _ema(close, period)
    e2 = _ema(e1, period)
    e3 = _ema(e2, period)
    # Rate of change: (current - previous) / previous
    trix_raw = np.diff(e3, prepend=e3[0])
    trix = trix_raw / np.where(e3[:-1] == 0, 1e-10, e3[:-1])
    trix = np.append(trix, 0)  # pad to same length
    return trix

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d TRIX for trend filter (daily timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    trix_1d = _trix(close_1d, 12)  # Standard TRIX period
    trix_1d_smooth = _ema(trix_1d, 9)  # Signal line smoothing
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d_smooth)
    
    # 1d Volume spike confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = _ema(vol_1d, 20)
    vol_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # 4h ATR for volatility filter and stop management
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = _ema(tr, atr_period)
    
    # 4h EMA for dynamic trend filter
    ema_fast = _ema(close, 9)
    ema_slow = _ema(close, 21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if TRIX data not ready
        if np.isnan(trix_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility periods
        if atr[i] < 0.01 * close[i]:  # Less than 1% ATR
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. Daily TRIX above signal line (bullish momentum)
            # 2. Daily volume spike (institutional interest)
            # 3. 4h fast EMA above slow EMA (short-term trend alignment)
            if (trix_1d_aligned[i] > 0) and vol_spike_aligned[i] and (ema_fast[i] > ema_slow[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Daily TRIX below zero (bearish momentum)
            # 2. Daily volume spike
            # 3. 4h fast EMA below slow EMA
            elif (trix_1d_aligned[i] < 0) and vol_spike_aligned[i] and (ema_fast[i] < ema_slow[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX turns bearish OR EMA crossover breaks down
            if (trix_1d_aligned[i] < 0) or (ema_fast[i] < ema_slow[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX turns bullish OR EMA crossover breaks up
            if (trix_1d_aligned[i] > 0) or (ema_fast[i] > ema_slow[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals