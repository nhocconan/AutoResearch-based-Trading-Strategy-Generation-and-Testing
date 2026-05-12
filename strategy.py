#!/usr/bin/env python3
"""
12h_TRIX_Trend_Confirm_Volume_Spike
Strategy: TRIX (15,9) signal line cross with 1w EMA200 trend filter and 1d volume spike
- TRIX(15,9) crosses above signal line = momentum bullish, below = bearish
- 1w EMA200 for long-term trend filter (only long above, short below)
- 1d volume > 2.0x 20-period average for confirmation
- Position size: 0.25
- Exit: reverse TRIX signal or trend filter fails
- Designed for fewer trades (<150 total over 4 years) to avoid fee drag
"""

name = "12h_TRIX_Trend_Confirm_Volume_Spike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX: TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX value
    trix = pd.Series(ema3).pct_change() * 100
    trix = trix.fillna(0).values
    
    # TRIX signal line: EMA of TRIX, 9-period
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align all indicators to 12h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)  # volume spike already 1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        if (np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(trix_aligned[i]) or
            np.isnan(trix_signal_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above signal line + price above weekly EMA200 + volume spike
            if (trix_aligned[i] > trix_signal_aligned[i] and
                trix_aligned[i-1] <= trix_signal_aligned[i-1] and
                close[i] > ema_200_1w_aligned[i] and
                volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below signal line + price below weekly EMA200 + volume spike
            elif (trix_aligned[i] < trix_signal_aligned[i] and
                  trix_aligned[i-1] >= trix_signal_aligned[i-1] and
                  close[i] < ema_200_1w_aligned[i] and
                  volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below signal line OR price below weekly EMA200
            if (trix_aligned[i] < trix_signal_aligned[i] and
                trix_aligned[i-1] >= trix_signal_aligned[i-1]) or \
               (close[i] < ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above signal line OR price above weekly EMA200
            if (trix_aligned[i] > trix_signal_aligned[i] and
                trix_aligned[i-1] <= trix_signal_aligned[i-1]) or \
               (close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals