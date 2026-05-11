#!/usr/bin/env python3
name = "1d_KAMA_RSI_ChopFilter"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # KAMA on daily close
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, n=10, prepend=close[:10]))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6 - 0.06) + 0.06) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh - ll) != 0, 100 * np.log10(np.nansum(atr) / (hh - ll)) / np.log10(14), 50)
    
    # Align weekly KAMA trend
    kama_1w = np.zeros_like(close_1w)
    change_1w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility_1w = np.abs(np.diff(close_1w, n=10, prepend=close_1w[:10]))
    er_1w = np.where(volatility_1w != 0, change_1w / volatility_1w, 0)
    sc_1w = (er_1w * (0.6 - 0.06) + 0.06) ** 2
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc_1w[i] * (close_1w[i] - kama_1w[i-1])
    kama_1w_trend = kama_1w > np.roll(kama_1w, 1)
    kama_1w_trend_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(14, 10)
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(kama_1w_trend_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up (trend) + RSI > 50 + Chop < 61.8 (trending market)
            if kama[i] > kama[i-1] and rsi[i] > 50 and chop[i] < 61.8 and kama_1w_trend_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down (trend) + RSI < 50 + Chop < 61.8 (trending market)
            elif kama[i] < kama[i-1] and rsi[i] < 50 and chop[i] < 61.8 and kama_1w_trend_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down OR RSI < 40
            if kama[i] < kama[i-1] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up OR RSI > 60
            if kama[i] > kama[i-1] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals