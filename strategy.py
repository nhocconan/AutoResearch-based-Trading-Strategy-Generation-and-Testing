#!/usr/bin/env python3
"""
4h_1w_TRIX_VolumeSpike_Trend_v1
Hypothesis: TRIX momentum on 1-week timeframe combined with volume spike and 4h trend filter to capture strong moves in both bull and bear markets. Uses weekly TRIX to filter for strong momentum, volume spike for confirmation, and 4h EMA for trend alignment. Designed to reduce trade frequency while capturing significant trends.
"""

name = "4h_1w_TRIX_VolumeSpike_Trend_v1"
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
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1w data for TRIX and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate TRIX (15,9,9) - triple smoothed ROC
    close_1w = df_1w['close'].values
    # First smoothing: EMA15
    ema1 = pd.Series(close_1w).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Second smoothing: EMA9 of ema1
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    # Third smoothing: EMA9 of ema2
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    # TRIX: 100 * (ema3 - previous ema3) / previous ema3
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = 100 * (ema3[1:] - ema3[:-1]) / ema3[:-1]
    # Signal line: EMA9 of TRIX
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX signal to 4h timeframe
    trix_signal_aligned = align_htf_to_ltf(prices, df_1w, trix_signal)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(trix_signal_aligned[i]) or
            np.isnan(ema_50_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX above signal line + volume spike + price above EMA50
            if (trix_signal_aligned[i] > 0 and 
                volume_spike[i] and 
                close[i] > ema_50_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX below signal line + volume spike + price below EMA50
            elif (trix_signal_aligned[i] < 0 and 
                  volume_spike[i] and 
                  close[i] < ema_50_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below signal line OR price below EMA50
            if trix_signal_aligned[i] < 0 or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above signal line OR price above EMA50
            if trix_signal_aligned[i] > 0 or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals