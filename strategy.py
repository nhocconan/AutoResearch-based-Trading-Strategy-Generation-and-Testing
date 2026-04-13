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
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # EMA200 on 4h close
    close_4h = df_4h['close'].values
    ema200_4h = np.full(len(close_4h), np.nan)
    for i in range(200, len(close_4h)):
        ema200_4h[i] = np.mean(close_4h[i-200:i])
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:i+1])
            avg_loss[i] = np.mean(loss[1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20
    
    for i in range(100, n):
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
            
        if np.isnan(ema200_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Trend filter: price above/below EMA200 on 4h
        uptrend = close[i] > ema200_4h_aligned[i]
        downtrend = close[i] < ema200_4h_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Entry logic
        if uptrend and rsi_oversold and position != 1:
            position = 1
            signals[i] = position_size
        elif downtrend and rsi_overbought and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and (not uptrend or rsi_1d_aligned[i] > 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not downtrend or rsi_1d_aligned[i] < 50):
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_ema200_rsi_filter"
timeframe = "1h"
leverage = 1.0