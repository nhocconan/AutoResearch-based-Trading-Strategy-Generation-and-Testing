#!/usr/bin/env python3
# 4h_TRIX_1d_Trend_Volume
# Hypothesis: Uses TRIX (15-period) on 4h for momentum, 1d EMA200 for trend filter, and volume spikes for entry confirmation.
# Long when TRIX crosses above zero, price above 1d EMA200, and volume spike.
# Short when TRIX crosses below zero, price below 1d EMA200, and volume spike.
# Designed for low trade frequency (<100 total trades over 4 years) to minimize fee drift.
# Works in bull/bear markets by following 1d trend while using TRIX zero-cross for precise entries.

name = "4h_TRIX_1d_Trend_Volume"
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
    
    # TRIX on 4h: 15-period EMA applied three times, then percent change
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100
    trix_values = trix.values
    
    # Volume spike: >2.0x 20-period average (to reduce frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), trix_values)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(trix_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above zero + price above 1d EMA200 + volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and
                close[i] > ema_200_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + price below 1d EMA200 + volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and
                  close[i] < ema_200_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR price closes below 1d EMA200
            if (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0) or \
               (close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR price closes above 1d EMA200
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0) or \
               (close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals