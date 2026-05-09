#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_TRIX_ZeroCross_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX (15,9,9) on daily close
    close_1d = df_1d['close'].values
    # EMA1: 15-period
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2: 9-period of EMA1
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    # EMA3: 9-period of EMA2
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    # TRIX: 9-period EMA of EMA3, expressed as percentage change
    ema4 = pd.Series(ema3).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = np.zeros_like(ema4)
    trix[1:] = (ema4[1:] - ema4[:-1]) / ema4[:-1] * 100
    
    # TRIX signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX and signal to 6h
    trix_6h = align_htf_to_ltf(prices, df_1d, trix)
    trix_signal_6h = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_6h[i]) or np.isnan(trix_signal_6h[i]) or 
            np.isnan(ema34_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: TRIX crosses above signal line with uptrend and volume spike
            if trix_6h[i] > trix_signal_6h[i] and trix_6h[i-1] <= trix_signal_6h[i-1] and \
               close[i] > ema34_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line with downtrend and volume spike
            elif trix_6h[i] < trix_signal_6h[i] and trix_6h[i-1] >= trix_signal_6h[i-1] and \
                 close[i] < ema34_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below signal line OR trend turns down
            if trix_6h[i] < trix_signal_6h[i] and trix_6h[i-1] >= trix_signal_6h[i-1] or \
               close[i] < ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above signal line OR trend turns up
            if trix_6h[i] > trix_signal_6h[i] and trix_6h[i-1] <= trix_signal_6h[i-1] or \
               close[i] > ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals