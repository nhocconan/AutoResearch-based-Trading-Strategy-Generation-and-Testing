#!/usr/bin/env python3
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
    
    # Get daily data for 12h context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 12-period RSI for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(12, n):
        if i == 12:
            avg_gain[i] = np.mean(gain[1:13])
            avg_loss[i] = np.mean(loss[1:13])
        else:
            avg_gain[i] = (avg_gain[i-1] * 11 + gain[i]) / 12
            avg_loss[i] = (avg_loss[i-1] * 11 + loss[i]) / 12
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_12 = np.full(n, np.nan)
    rsi_12[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    # Get weekly data for trend filter: EMA(50) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_50 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i < 49:
            ema_1w_50[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_50[i-1]):
                ema_1w_50[i] = np.mean(close_1w[i-49:i+1])
            else:
                ema_1w_50[i] = close_1w[i] * alpha_w + ema_1w_50[i-1] * (1 - alpha_w)
    
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(12, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(rsi_12[i]) or
            np.isnan(ema_1w_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) + weekly uptrend
            if (rsi_12[i] < 30 and 
                ema_1w_50_aligned[i] > ema_1w_50_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + weekly downtrend
            elif (rsi_12[i] > 70 and 
                  ema_1w_50_aligned[i] < ema_1w_50_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 or weekly trend turns down
            if (rsi_12[i] > 70 or 
                ema_1w_50_aligned[i] < ema_1w_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 or weekly trend turns up
            if (rsi_12[i] < 30 or 
                ema_1w_50_aligned[i] > ema_1w_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI12_WeeklyEMA50_v1"
timeframe = "12h"
leverage = 1.0