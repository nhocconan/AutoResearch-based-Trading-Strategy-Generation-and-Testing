#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h ATR(14) for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # 4h Close for trend filter (Hull Moving Average 21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_4h).ewm(span=half_len, adjust=False).mean()
    wma_full = pd.Series(close_4h).ewm(span=21, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
    hma_21_aligned = align_htf_to_ltf(prices, df_4h, hma_21)
    
    # 1d Close for regime filter (Close > SMA50 = bull, < SMA50 = bear)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Hour filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # Fixed 20% position
    
    start = max(14, 21, 50)  # ATR(14), HMA(21), SMA50(1d)
    for i in range(start, n):
        if (np.isnan(atr[i]) or np.isnan(hma_21_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price > HMA21(4h) AND price > SMA50(1d) AND volatility expansion (ATR rising)
            if (price > hma_21_aligned[i] and price > sma_50_1d_aligned[i] and 
                atr[i] > atr[i-1]):
                position = 1
                signals[i] = position_size
            # Short: price < HMA21(4h) AND price < SMA50(1d) AND volatility expansion
            elif (price < hma_21_aligned[i] and price < sma_50_1d_aligned[i] and 
                  atr[i] > atr[i-1]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < HMA21(4h) OR volatility contraction
            if price < hma_21_aligned[i] or atr[i] < atr[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price > HMA21(4h) OR volatility contraction
            if price > hma_21_aligned[i] or atr[i] < atr[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_HMA_SMA_Volatility_Filter"
timeframe = "1h"
leverage = 1.0