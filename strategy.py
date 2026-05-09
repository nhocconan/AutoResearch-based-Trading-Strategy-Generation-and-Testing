#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_TRIX_Signal_RSI_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    close_d = df_d['close'].values
    
    # Calculate TRIX(12): triple EMA of log returns
    # Step 1: log returns
    log_ret = np.diff(np.log(close_d), prepend=np.log(close_d[0]))
    # Step 2: triple EMA
    ema1 = pd.Series(log_ret).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = 100 * ema3  # TRIX value
    
    # Align TRIX to 12h timeframe
    trix_d_aligned = align_htf_to_ltf(prices, df_d, trix_raw)
    
    # Get weekly data for RSI filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 14:
        return np.zeros(n)
    
    close_w = df_w['close'].values
    delta = np.diff(close_w, prepend=close_w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi_w = 100 - (100 / (1 + rs))
    rsi_w_aligned = align_htf_to_ltf(prices, df_w, rsi_w)
    
    # Volume filter: current volume > 1.1 * 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma * 1.1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Need enough data for TRIX and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(trix_d_aligned[i]) or 
            np.isnan(rsi_w_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_d_aligned[i]
        rsi_val = rsi_w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: TRIX crosses above zero + RSI > 50 + volume filter
            if trix_val > 0 and rsi_val > 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero + RSI < 50 + volume filter
            elif trix_val < 0 and rsi_val < 50 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero or RSI < 40
            if trix_val < 0 or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero or RSI > 60
            if trix_val > 0 or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals