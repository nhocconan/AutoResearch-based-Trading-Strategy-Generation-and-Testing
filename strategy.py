#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend and chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA calculation on 1w close
    close_1w = df_1w['close'].values
    delta = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    direction = np.abs(close_1w - np.roll(close_1w, 10))
    volatility = np.sum(delta.reshape(-1, 10), axis=1)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Chop filter on 1w: EMA of True Range / ATR
    tr1 = high[1:] - low[:-1]
    tr2 = np.abs(high[1:] - np.roll(close_1w, 1)[1:])
    tr3 = np.abs(low[1:] - np.roll(close_1w, 1)[1:])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.insert(tr, 0, high[0] - low[0])
    atr = np.zeros_like(close_1w)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    atr_ma = pd.Series(atr).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(atr_ma / np.sum(tr.reshape(-1, 14), axis=1)) / np.log10(14)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align all to 12h
    kama_12h = align_htf_to_ltf(prices, df_1w, kama)
    chop_12h = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 14)  # Need RSI and chop data
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(rsi_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_12h[i]
        chop_val = chop_12h[i]
        rsi_val = rsi_12h[i]
        
        if position == 0:
            # Enter long: price > KAMA, RSI > 50, chop < 61.8 (trending)
            if close[i] > kama_val and rsi_val > 50 and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA, RSI < 50, chop < 61.8 (trending)
            elif close[i] < kama_val and rsi_val < 50 and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA or RSI < 40 or chop > 61.8 (choppy)
            if close[i] < kama_val or rsi_val < 40 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA or RSI > 60 or chop > 61.8 (choppy)
            if close[i] > kama_val or rsi_val > 60 or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals