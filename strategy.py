#!/usr/bin/env python3
name = "1h_TRIX_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for TRIX calculation (trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 15:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # TRIX(12): triple EMA of percent change
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    pct_change = np.diff(ema3, prepend=ema3[0]) / (ema3[:-1] + 1e-10) * 100
    pct_change = np.append(pct_change[0], pct_change)  # align length
    trix = pd.Series(pct_change).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_signal = trix > 0  # positive TRIX = bullish momentum
    
    trix_signal_aligned = align_htf_to_ltf(prices, df_4h, trix_signal)
    
    # Get 1d data for volume filter (volume spike)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume_1d > 1.5 * vol_ma20
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # 1h volume confirmation
    vol_ma20_1h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > 1.2 * vol_ma20_1h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trix_signal_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: TRIX bullish + daily volume spike + session + 1h volume
        if position == 0:
            if trix_signal_aligned[i] and volume_spike_aligned[i] and session_filter[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Entry conditions: TRIX bearish + daily volume spike + session + 1h volume
            elif not trix_signal_aligned[i] and volume_spike_aligned[i] and session_filter[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: TRIX turns bearish OR session ends
            if not trix_signal_aligned[i] or not session_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: TRIX turns bullish OR session ends
            if trix_signal_aligned[i] or not session_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals