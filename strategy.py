#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX and chop filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    close_d = df_d['close'].values
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    
    # Calculate TRIX (15-period EMA of EMA of EMA)
    close_series = pd.Series(close_d)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.diff() / ema3.shift(1))
    trix = trix.replace([np.inf, -np.inf], np.nan).fillna(0).values
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate Choppiness Index (14-period)
    atr_d = []
    tr_d = []
    for i in range(len(high_d)):
        if i == 0:
            tr = high_d[i] - low_d[i]
        else:
            tr = max(high_d[i] - low_d[i], abs(high_d[i] - close_d[i-1]), abs(low_d[i] - close_d[i-1]))
        tr_d.append(tr)
        atr_d.append(np.nan)
    
    tr_d = np.array(tr_d)
    atr_series = pd.Series(tr_d)
    atr14 = atr_series.rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high_d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_d).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr14.sum() / (max_high - min_low)) / np.log10(14)
    chop = np.where((max_high - min_low) == 0, 50, chop)
    
    # Align TRIX signal and chop to 4h
    trix_signal_aligned = align_htf_to_ltf(prices, df_d, trix_signal)
    chop_aligned = align_htf_to_ltf(prices, df_d, chop)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_signal_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        chop_ok = chop_aligned[i] > 61.8  # choppy/range market
        
        if position == 0:
            # Long: TRIX crosses above signal line in choppy market with volume
            if trix[i] > trix_signal_aligned[i] and trix[i-1] <= trix_signal_aligned[i-1] and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line in choppy market with volume
            elif trix[i] < trix_signal_aligned[i] and trix[i-1] >= trix_signal_aligned[i-1] and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below signal line
            if trix[i] < trix_signal_aligned[i] and trix[i-1] >= trix_signal_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above signal line
            if trix[i] > trix_signal_aligned[i] and trix[i-1] <= trix_signal_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals