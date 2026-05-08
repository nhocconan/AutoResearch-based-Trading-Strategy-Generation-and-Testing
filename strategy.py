#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_TRIX_ZeroLine_Cross_1wTrend_Volume"
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
    
    # TRIX calculation: EMA of EMA of EMA of log returns
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    # Log returns
    log_returns = np.diff(np.log(close), prepend=np.log(close[0]))
    # Triple EMA
    ema1 = ema(log_returns, 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)
    trix = 100 * (ema3 / np.roll(ema3, 1) - 1)
    trix[0] = 0  # First value has no previous
    
    # 1-week trend: EMA50 on weekly closes
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = ema(close_1w, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for TRIX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero + uptrend (price > 1w EMA50) + volume spike
            long_cond = (trix[i-1] <= 0) and (trix[i] > 0) and \
                        (close[i] > ema_50_1w_aligned[i]) and \
                        volume_spike[i]
            # Short: TRIX crosses below zero + downtrend (price < 1w EMA50) + volume spike
            short_cond = (trix[i-1] >= 0) and (trix[i] < 0) and \
                         (close[i] < ema_50_1w_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals