#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_0_Signal_ZeroLag_Trend"
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
    
    # 4h TRIX (15-period EMA smoothed 3x) for momentum
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3.pct_change() * 100).values  # TRIX in percentage
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d volume filter: current 4h volume > 1.3 * 20-period average of 4h volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(15+15+15, 20, 9)  # TRIX smoothing + volume MA + signal line
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix[i]
        trix_sig = trix_signal[i]
        ema50_val = ema50_12h_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: TRIX crosses above signal + above 12h EMA50 + volume filter
            if trix_val > trix_sig and trix_val > 0 and close[i] > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below signal + below 12h EMA50 + volume filter
            elif trix_val < trix_sig and trix_val < 0 and close[i] < ema50_val and vol_filter:
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