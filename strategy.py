#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Trend_RSI_1d"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h KAMA
    price_series = pd.Series(close)
    delta = price_series.diff().abs()
    direction = abs(price_series - price_series.shift(10))
    volatility = delta.rolling(window=10, min_periods=10).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    rsi_period = 14
    delta_1d = df_1d['close'].diff()
    gain = (delta_1d.where(delta_1d > 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    loss = (-delta_1d.where(delta_1d < 0, 0)).rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Align 1d RSI to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Align KAMA to 12h (already in same timeframe, but ensure alignment)
    kama_aligned = align_htf_to_ltf(prices, prices, kama)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, rsi_period)
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Enter long: price above KAMA and RSI < 50 (bullish momentum in bear)
            if close[i] > kama_val and rsi_val < 50:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA and RSI > 50 (bearish momentum in bull)
            elif close[i] < kama_val and rsi_val > 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA or RSI > 70 (overbought)
            if close[i] < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA or RSI < 30 (oversold)
            if close[i] > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals