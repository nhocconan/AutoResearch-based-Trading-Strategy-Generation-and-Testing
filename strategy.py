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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14_1d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 14-day RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(len(df_1d), np.nan)
    avg_loss = np.full(len(df_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(len(df_1d), np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_14 = np.full(len(df_1d), np.nan)
    rsi_14[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate daily 50-period EMA
    ema_50_1d = np.full(len(df_1d), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_1d)):
        if i < 50:
            ema_50_1d[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_50_1d[i-1]):
                ema_50_1d[i] = np.mean(close_1d[i-49:i+1])
            else:
                ema_50_1d[i] = close_1d[i] * alpha + ema_50_1d[i-1] * (1 - alpha)
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price above daily EMA50
            if (rsi_14_aligned[i] < 30 and 
                price > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + price below daily EMA50
            elif (rsi_14_aligned[i] > 70 and 
                  price < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 50 or price below daily EMA50
            if (rsi_14_aligned[i] > 50 or 
                price < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 50 or price above daily EMA50
            if (rsi_14_aligned[i] < 50 or 
                price > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyRSI_EMA50_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0