#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and volatility filters
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 30:
        return np.zeros(n)
    
    # Calculate 50-period EMA on daily close for trend filter
    close_d = df_d['close'].values
    ema50_d = np.full(len(close_d), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_d)):
        if i < 50:
            ema50_d[i] = np.nan
        elif i == 50:
            ema50_d[i] = np.mean(close_d[:50])
        else:
            ema50_d[i] = alpha * close_d[i] + (1 - alpha) * ema50_d[i-1]
    
    # Align daily EMA50 to 6h
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    
    # Calculate 14-period ATR on daily for volatility filter
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d_arr = df_d['close'].values
    tr_d = np.maximum(high_d[1:] - low_d[1:], 
                      np.maximum(np.abs(high_d[1:] - close_d_arr[:-1]), 
                                 np.abs(low_d[1:] - close_d_arr[:-1])))
    tr_d = np.concatenate([[np.nan], tr_d])
    atr_d = np.full(len(tr_d), np.nan)
    for i in range(14, len(tr_d)):
        if i == 14:
            atr_d[i] = np.mean(tr_d[1:15])
        else:
            atr_d[i] = (atr_d[i-1] * 13 + tr_d[i]) / 14
    
    # Align daily ATR14 to 6h
    atr_d_aligned = align_htf_to_ltf(prices, df_d, atr_d)
    
    # Calculate 6-period RSI on 6h for entry timing
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(6, n):
        if i == 6:
            avg_gain[i] = np.mean(gain[1:7])
            avg_loss[i] = np.mean(loss[1:7])
        else:
            avg_gain[i] = (avg_gain[i-1] * 5 + gain[i]) / 6
            avg_loss[i] = (avg_loss[i-1] * 5 + loss[i]) / 6
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, 14, 6) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_d_aligned[i]) or np.isnan(atr_d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50 = ema50_d_aligned[i]
        atr = atr_d_aligned[i]
        
        if position == 0:
            # Long: Price above daily EMA50 AND RSI oversold (<30)
            if price > ema50 and rsi[i] < 30:
                signals[i] = size
                position = 1
            # Short: Price below daily EMA50 AND RSI overbought (>70)
            elif price < ema50 and rsi[i] > 70:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below daily EMA50 OR RSI overbought (>70)
            if price < ema50 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above daily EMA50 OR RSI oversold (<30)
            if price > ema50 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_EMA50_RSI_Filter"
timeframe = "6h"
leverage = 1.0