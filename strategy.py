#!/usr/bin/env python3
name = "4h_KAMA_Direction_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chop (Choppiness Index) on 1d
    atr_1d = np.zeros(len(close_1d))
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d[1:] = np.where(np.arange(1, len(tr)) < 14, 
                          np.mean(tr[:14]) if len(tr) >= 14 else np.mean(tr) if len(tr) > 0 else 0,
                          0)
    for i in range(14, len(atr_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i-1]) / 14
    
    sum_atr_14 = np.zeros(len(close_1d))
    for i in range(13, len(close_1d)):
        sum_atr_14[i] = np.sum(atr_1d[i-13:i+1])
    
    hh = np.maximum.accumulate(high_1d)
    ll = np.minimum.accumulate(low_1d)
    range_14 = hh - ll
    chop = np.zeros(len(close_1d))
    for i in range(13, len(close_1d)):
        if sum_atr_14[i] > 0 and range_14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr_14[i] / range_14[i]) / np.log10(14)
        else:
            chop[i] = 50
    
    chop_filter = chop < 61.8  # Trending regime (chop < 61.8)
    chop_filter = np.where(np.isnan(chop_filter), False, chop_filter)
    
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # Calculate KAMA on 4h
    def kama(price, period=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / volatility[period-1:]
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama_val = kama(close, 10, 2, 30)
    kama_diff = kama_val - np.roll(kama_val, 1)
    kama_diff[0] = 0
    kama_up = kama_diff > 0
    
    # Calculate RSI on 4h
    def rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_val = rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_up[i]) or np.isnan(rsi_val[i]) or np.isnan(chop_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + Chop < 61.8 (trending)
            if kama_up[i] and rsi_val[i] > 50 and chop_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + Chop < 61.8 (trending)
            elif not kama_up[i] and rsi_val[i] < 50 and chop_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down or RSI < 40
            if not kama_up[i] or rsi_val[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up or RSI > 60
            if kama_up[i] or rsi_val[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals