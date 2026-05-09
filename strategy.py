#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Trix_Trend_Volume_Filter_v2"
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
    
    # Get 1d data for TRIX and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate TRIX (15-period EMA triple smoothed)
    close_1d = pd.Series(df_1d['close'])
    ema1 = close_1d.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ((ema3 / ema3.shift(1)) - 1) * 100
    trix_values = trix.values
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 4h volume > 1.5 * 20-period average
    vol_series = pd.Series(df_4h['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_4h = df_4h['volume'].values > (vol_ma * 1.5)
    
    # Align all to 4h
    trix_4h = align_htf_to_ltf(prices, df_1d, trix_values)
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_filter_4h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(trix_4h[i]) or np.isnan(ema50_1d_4h[i]) or
            np.isnan(volume_filter_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_4h[i]
        trend = ema50_1d_4h[i]
        vol_filter = volume_filter_4h_aligned[i]
        
        if position == 0:
            # Enter long: TRIX crosses above zero with volume and above trend
            if trix_val > 0 and trix_4h[i-1] <= 0 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero with volume and below trend
            elif trix_val < 0 and trix_4h[i-1] >= 0 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_val < 0 and trix_4h[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_val > 0 and trix_4h[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals