#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_TRIX_Signal_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # Get daily data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # TRIX calculation on daily close (15-period EMA triple)
    close_1d = pd.Series(df_1d['close'])
    ema1 = close_1d.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix_signal = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    trix_hist = trix - trix_signal
    
    # Weekly trend filter: price above/below weekly EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Volume filter: current 12h volume > 1.3 * 20-period average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    vol_series = pd.Series(df_12h['volume'])
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    volume_filter_12h = df_12h['volume'].values > (vol_ma * 1.3)
    
    # Align all to 12h
    trix_hist_12h = align_htf_to_ltf(prices, df_1d, trix_hist.values)
    ema50_1w_12h = align_htf_to_ltf(prices, df_1w, ema50_1w.values)
    volume_filter_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_filter_12h.values)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data
    
    for i in range(start_idx, n):
        if (np.isnan(trix_hist_12h[i]) or np.isnan(ema50_1w_12h[i]) or 
            np.isnan(volume_filter_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_h = trix_hist_12h[i]
        trend = ema50_1w_12h[i]
        vol_filter = volume_filter_12h_aligned[i]
        
        if position == 0:
            # Enter long: TRIX histogram crosses above zero with uptrend and volume
            if trix_h > 0 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX histogram crosses below zero with downtrend and volume
            elif trix_h < 0 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX histogram crosses below zero
            if trix_h < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX histogram crosses above zero
            if trix_h > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals