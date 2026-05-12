#!/usr/bin/env python3
"""
4h_TRIX_18_Signal_Line_Cross_1dTrend_VolumeSpike
- TRIX(18) crossing above/below its 9-period signal line (zero-lag smoothed momentum)
- Trend filter: price above/below 1-day EMA34
- Volume confirmation: >1.3x 20-period average volume
- Exit: opposite TRIX signal line cross or price crosses 1-day EMA34
- Position size: 0.25 to limit drawdown
- Target: ~100-150 trades over 4 years to avoid excessive fee drag
"""

name = "4h_TRIX_18_Signal_Line_Cross_1dTrend_VolumeSpike"
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
    
    # Volume spike: >1.3x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # TRIX indicator: triple-smoothed EMA of ROC, then signal line
    # ROC = (close - close.shift(1)) / close.shift(1) * 100
    roc = np.diff(close, prepend=close[0]) / close
    # EMA1 of ROC
    ema1 = pd.Series(roc).ewm(span=18, adjust=False, min_periods=18).mean().values
    # EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=18, adjust=False, min_periods=18).mean().values
    # EMA3 of EMA2 = TRIX
    ema3 = pd.Series(ema2).ewm(span=18, adjust=False, min_periods=18).mean().values
    # Signal line: 9-period EMA of TRIX
    signal_line = pd.Series(ema3).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # 1-day data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Align TRIX and signal line to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema3)
    signal_line_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), signal_line)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(trix_aligned[i]) or
            np.isnan(signal_line_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above signal line + price above 1d EMA34 + volume spike
            if (trix_aligned[i] > signal_line_aligned[i] and
                trix_aligned[i-1] <= signal_line_aligned[i-1] and
                close[i] > ema_34_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below signal line + price below 1d EMA34 + volume spike
            elif (trix_aligned[i] < signal_line_aligned[i] and
                  trix_aligned[i-1] >= signal_line_aligned[i-1] and
                  close[i] < ema_34_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below signal line OR price crosses below 1d EMA34
            if (trix_aligned[i] < signal_line_aligned[i] and
                trix_aligned[i-1] >= signal_line_aligned[i-1]) or \
               (close[i] < ema_34_1d_aligned[i] and
                close[i-1] >= ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above signal line OR price crosses above 1d EMA34
            if (trix_aligned[i] > signal_line_aligned[i] and
                trix_aligned[i-1] <= signal_line_aligned[i-1]) or \
               (close[i] > ema_34_1d_aligned[i] and
                close[i-1] <= ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals