#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_TRIX_Volume_Spike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # TRIX calculation: EMA(EMA(EMA(close,12),12),12)
    close_series = pd.Series(df_1d['close'])
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    
    # TRIX = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix_raw = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix_raw.values
    
    # TRIX signal line: 9-period EMA of TRIX
    trix_series = pd.Series(trix)
    trix_signal = trix_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX and signal line to daily data (already aligned as we're using 1d timeframe)
    trix_aligned = trix  # Already at 1d frequency
    trix_signal_aligned = trix_signal  # Already at 1d frequency
    
    # Trend filter: 50-period EMA on daily close
    ema50_1d = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 2.0 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 50)  # Need enough data for TRIX and EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or
            np.isnan(ema50_1d[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_aligned[i]
        trix_signal_val = trix_signal_aligned[i]
        trend = ema50_1d[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: TRIX crosses above signal line with volume and above trend
            if trix_val > trix_signal_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below signal line with volume and below trend
            elif trix_val < trix_signal_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below signal line
            if trix_val < trix_signal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above signal line
            if trix_val > trix_signal_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals