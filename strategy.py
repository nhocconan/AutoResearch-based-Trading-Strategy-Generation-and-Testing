#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_TRIX_Trend_Volume_Filter_v3"
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
    
    # Get 1d data for TRIX and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # TRIX calculation: TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    close_series = pd.Series(df_1d['close'])
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # Percentage change
    
    # Trend filter: 1d EMA50
    ema50_1d = close_series.ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Volume filter: current 6h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (vol_ma * 1.5)
    
    # Align all to 6h (primary timeframe)
    trix_6h = align_htf_to_ltf(prices, df_1d, trix.values)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d.values)
    vol_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter.values)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(45, 20)  # Need enough data for TRIX (3*15) and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(trix_6h[i]) or np.isnan(ema50_1d_6h[i]) or 
            np.isnan(vol_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_6h[i]
        trend = ema50_1d_6h[i]
        vol_filter = vol_filter_6h[i]
        
        if position == 0:
            # Enter long: TRIX crosses above zero with volume and above trend
            if trix_val > 0 and trix_6h[i-1] <= 0 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero with volume and below trend
            elif trix_val < 0 and trix_6h[i-1] >= 0 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_val < 0 and trix_6h[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_val > 0 and trix_6h[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals