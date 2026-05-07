#!/usr/bin/env python3
name = "1d_KAMA_Trend_With_RSI_Momentum"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d KAMA for trend direction
    def kama(close_series, er_len=10, fast_len=2, slow_len=30):
        close_s = pd.Series(close_series)
        change = abs(close_s.diff(er_len))
        volatility = close_s.diff().abs().rolling(er_len).sum()
        er = change / volatility.replace(0, np.nan)
        er = er.fillna(0)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama_vals = np.zeros_like(close_s, dtype=np.float64)
        kama_vals[0] = close_s.iloc[0]
        for i in range(1, len(close_s)):
            kama_vals[i] = kama_vals[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    
    # 1d RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 1w KAMA for higher timeframe trend filter
    kama_1w = kama(df_1w['close'].values, 10, 2, 30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Volume filter: above 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for KAMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, volume filter, price above KAMA
            if kama_vals[i] > kama_vals[i-1] and rsi[i] > 50 and close[i] > kama_vals[i] and vol_filter[i] and close[i] > kama_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, volume filter, price below KAMA
            elif kama_vals[i] < kama_vals[i-1] and rsi[i] < 50 and close[i] < kama_vals[i] and vol_filter[i] and close[i] < kama_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA turns down or RSI < 40
            if kama_vals[i] < kama_vals[i-1] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA turns up or RSI > 60
            if kama_vals[i] > kama_vals[i-1] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals