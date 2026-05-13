#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Reversal_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA parameters
    er_len = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    # Recalculate volatility properly
    volatility = np.zeros(n)
    for i in range(n):
        if i == 0:
            volatility[i] = 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-er_len+1):i+1])))
    change_sum = np.zeros(n)
    for i in range(n):
        change_sum[i] = np.sum(np.abs(np.diff(close[max(0, i-er_len+1):i+1])))
    er = np.where(change_sum != 0, change / change_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(n):
        if i < 14:
            avg_gain[i] = np.mean(gain[max(0, i-13):i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[max(0, i-13):i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(tr1, np.maximum(tr2, tr3))
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[max(0, i-13):i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of true range over 14 periods
    sum_tr = np.zeros(n)
    for i in range(n):
        if i < 14:
            sum_tr[i] = np.sum(tr[max(0, i-13):i+1])
        else:
            sum_tr[i] = sum_tr[i-1] - tr[i-14] + tr[i]
    
    # Highest high and lowest low over 14 periods
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(n):
        if i < 14:
            max_high[i] = np.max(high[max(0, i-13):i+1])
            min_low[i] = np.min(low[max(0, i-13):i+1])
        else:
            max_high[i] = np.max(high[i-13:i+1])
            min_low[i] = np.min(low[i-13:i+1])
    
    chop = np.where((max_high - min_low) != 0, 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(14), 50)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > KAMA, RSI < 30 (oversold), Chop > 61.8 (ranging), Weekly uptrend
            if close[i] > kama[i] and rsi[i] < 30 and chop[i] > 61.8 and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA, RSI > 70 (overbought), Chop > 61.8 (ranging), Weekly downtrend
            elif close[i] < kama[i] and rsi[i] > 70 and chop[i] > 61.8 and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA OR RSI > 70 (overbought)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA OR RSI < 30 (oversold)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals