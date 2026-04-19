#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Camarilla_Reverse_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for multi-timeframe analysis
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate previous day's Camarilla levels
    prev_close_4h = np.concatenate([[np.nan], close_4h[:-1]])
    prev_high_4h = np.concatenate([[np.nan], high_4h[:-1]])
    prev_low_4h = np.concatenate([[np.nan], low_4h[:-1]])
    
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    range_4h = prev_high_4h - prev_low_4h
    
    # Camarilla levels: R1, R2, S1, S2
    r1_4h = pivot_4h + range_4h * 1.1 / 12
    r2_4h = pivot_4h + range_4h * 1.1 / 6
    s1_4h = pivot_4h - range_4h * 1.1 / 12
    s2_4h = pivot_4h - range_4h * 1.1 / 6
    
    # Align to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    
    # 1h RSI for overbought/oversold
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or \
           np.isnan(r2_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or \
           np.isnan(s2_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hour = hours[i]
        
        # Session filter: only trade 8-20 UTC
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility periods
        if atr_1d_aligned[i] < 0.01 * price:  # Less than 1% ATR
            signals[i] = 0.0
            continue
        
        # Trend bias from 1d EMA200
        long_bias = price > ema200_1d_aligned[i]
        short_bias = price < ema200_1d_aligned[i]
        
        if position == 0:
            # Long: price rejects S1/S2 support with RSI oversold
            if (price > s1_4h_aligned[i] and 
                low[i] <= s2_4h_aligned[i] and 
                rsi[i] < 30 and 
                long_bias):
                signals[i] = 0.20
                position = 1
            # Short: price rejects R1/R2 resistance with RSI overbought
            elif (price < r1_4h_aligned[i] and 
                  high[i] >= r2_4h_aligned[i] and 
                  rsi[i] > 70 and 
                  short_bias):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price reaches pivot or RSI overbought
            if price >= pivot_4h_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price reaches pivot or RSI oversold
            if price <= pivot_4h_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals