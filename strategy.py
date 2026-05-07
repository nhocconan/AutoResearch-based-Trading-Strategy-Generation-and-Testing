#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Trend_v1"
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
    
    # 1d KAMA (adaptive moving average) for trend direction
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array lengths
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 14-period RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period) for regime filter
    atr = np.full_like(close, np.nan)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    for i in range(1, n):
        if i < 14:
            atr[i] = np.nan
        else:
            if np.isnan(atr[i-1]):
                atr[i] = np.nanmean(tr[i-13:i+1])
            else:
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # Sum of true range over 14 periods
    tr_sum = np.full_like(close, np.nan)
    for i in range(13, n):
        tr_sum[i] = np.nansum(tr[i-13:i+1])
    # Highest high and lowest low over 14 periods
    max_high = np.full_like(close, np.nan)
    min_low = np.full_like(close, np.nan)
    for i in range(13, n):
        max_high[i] = np.nanmax(high[i-13:i+1])
        min_low[i] = np.nanmin(low[i-13:i+1])
    chop = np.where((max_high - min_low) != 0, 100 * np.log10(tr_sum / (max_high - min_low)) / np.log10(14), 50)
    
    # 1-week trend filter: EMA(34) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Warmup for KAMA, RSI, CHOP
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, Chop < 61.8 (trending), weekly uptrend
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and 
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50, Chop < 61.8 (trending), weekly downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and 
                  close[i] < ema_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or RSI < 40
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or RSI > 60
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals