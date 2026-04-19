#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Chop_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio (ER) for weekly data
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w, axis=0)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    er = np.concatenate([[0], er[1:]])  # Align with close_1w
    
    # Calculate Smoothing Constant (SC) and KAMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Weekly RSI
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly Chopiness Index
    atr_period = 14
    chop_period = 14
    tr1 = np.maximum(high_1w[1:], close_1w[:-1]) - np.minimum(low_1w[1:], close_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high_1w).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low_1w).rolling(window=chop_period, min_periods=chop_period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    range_high_low = max_high - min_low
    chop = np.where(range_high_low != 0, 100 * np.log10(sum_atr / range_high_low) / np.log10(chop_period), 50)
    
    # Align weekly indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Daily ATR for stop loss
    tr1_d = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr2_d = np.abs(high[1:] - close[:-1])
    tr3_d = np.abs(low[1:] - close[:-1])
    tr_d = np.concatenate([[np.nan], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 14)
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(atr_d[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_d[i]
        
        # Trend direction: price relative to KAMA
        bullish = price > kama_aligned[i]
        bearish = price < kama_aligned[i]
        
        # RSI conditions
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Chop regime: chop > 50 indicates ranging market
        ranging = chop_aligned[i] > 50
        
        if position == 0:
            # Long in ranging market when RSI oversold
            if rsi_oversold and ranging:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short in ranging market when RSI overbought
            elif rsi_overbought and ranging:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: RSI overbought or stop loss
            if rsi_overbought or price < entry_price - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: RSI oversold or stop loss
            if rsi_oversold or price > entry_price + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals