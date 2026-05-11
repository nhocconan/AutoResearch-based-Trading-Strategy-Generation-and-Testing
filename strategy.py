#!/usr/bin/env python3
name = "1d_KAMA_21_10_5_Signal_1wTrend_WeeklyVolume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly volume filter: volume > 1.5x 10-period average
    vol_ma_10_1w = pd.Series(df_1w['volume'].values).rolling(window=10, min_periods=10).mean().values
    vol_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10_1w)
    
    # KAMA parameters
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    kama_period = 5  # signal period
    
    # Calculate KAMA
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Fix volatility calculation
    volatility = np.array([np.sum(np.abs(np.diff(close[i:i+er_period]))) if i+er_period <= len(close) else 0 
                          for i in range(len(close))])
    volatility = np.concatenate([np.full(er_period-1, np.nan), volatility[er_period-1:]])
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    kama = np.full(n, np.nan)
    kama[er_period-1] = close[er_period-1]
    for i in range(er_period, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA signal: price relative to KAMA
    kama_signal = np.where(close > kama, 1, -1)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, er_period-1, 10)  # warmup period
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_10_1w_aligned[i]) or np.isnan(kama_signal[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA bullish + weekly uptrend + volume confirmation
            if (kama_signal[i] == 1 and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > vol_ma_10_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA bearish + weekly downtrend + volume confirmation
            elif (kama_signal[i] == -1 and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > vol_ma_10_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns bearish or weekly trend breaks
            if (kama_signal[i] == -1 or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns bullish or weekly trend breaks
            if (kama_signal[i] == 1 or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals