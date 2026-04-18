#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_TrendFilter
Hypothesis: 12h TRIX momentum oscillator with volume spike and 1d trend filter captures sustained moves in both bull and bear markets.
TRIX filters noise and identifies momentum shifts; volume spike confirms strength; 1d trend avoids counter-trend trades.
Designed for ~15-30 trades/year to minimize fee drag while capturing major trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX (15-period EMA applied 3 times)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3 * 100  # percentage change
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume spike: >2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Trend filter: 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(45, 30)  # Warmup for TRIX and volume
    
    for i in range(start_idx, n):
        if (np.isnan(trix_signal[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        t_signal = trix_signal[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: TRIX crosses above signal with volume spike and uptrend
            if i > 0 and trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1] and vol_spike and close[i] > ema34:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal with volume spike and downtrend
            elif i > 0 and trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1] and vol_spike and close[i] < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TRIX crosses below signal OR trend turns down
            if i > 0 and trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TRIX crosses above signal OR trend turns up
            if i > 0 and trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_TRIX_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0