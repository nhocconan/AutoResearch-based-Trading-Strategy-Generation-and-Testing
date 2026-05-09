#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_Trend_RSI_Overbought_Oversold"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA trend calculation
    close_series = pd.Series(df_1d['close'])
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = [close_series.iloc[0]]
    for i in range(1, len(close_series)):
        kama.append(kama[-1] + sc.iloc[i] * (close_series.iloc[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI calculation
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # Align KAMA and RSI to 12h
    kama_12h = align_htf_to_ltf(prices, df_1d, kama)
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need enough data for KAMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h[i]) or np.isnan(rsi_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_12h[i]
        rsi_val = rsi_12h[i]
        
        if position == 0:
            # Enter long: price above KAMA and RSI oversold (<30)
            if close[i] > kama_val and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA and RSI overbought (>70)
            elif close[i] < kama_val and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought (>70) or price below KAMA
            if rsi_val > 70 or close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold (<30) or price above KAMA
            if rsi_val < 30 or close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals