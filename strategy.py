#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_Volume_Spike_Chop_Filter_12hTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and TRIX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h TRIX (15-period EMA applied three times)
    close_12h = df_12h['close'].values
    ema1 = pd.Series(close_12h).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100  # Percentage change
    trix_signal = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.zeros(len(close_1d))
    atr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], 
                 abs(high_1d[i] - close_1d[i-1]), 
                 abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14
    
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum()
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    
    # 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    trix_signal_4h = align_htf_to_ltf(prices, df_12h, trix_signal.values)
    trix_4h = align_htf_to_ltf(prices, df_12h, trix.values)
    ema50_12h_4h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop.values)
    vol_avg_1d_4h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(trix_signal_4h[i]) or np.isnan(trix_4h[i]) or 
            np.isnan(ema50_12h_4h[i]) or np.isnan(chop_4h[i]) or 
            np.isnan(vol_avg_1d_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_4h[i]
        trix_sig = trix_signal_4h[i]
        trend = ema50_12h_4h[i]
        chop_val = chop_4h[i]
        vol_avg = vol_avg_1d_4h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        # Chop filter: only trade in ranging markets (Chop > 61.8)
        if chop_val <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal with volume
            if trix_val > trix_sig and trix_val > 0 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal with volume
            elif trix_val < trix_sig and trix_val < 0 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below signal
            if trix_val < trix_sig:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above signal
            if trix_val > trix_sig:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals