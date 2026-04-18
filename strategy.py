#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter
Hypothesis: KAMA adapts to market efficiency, reducing whipsaw in choppy markets. 
Combined with RSI for momentum confirmation and volatility filter for regime detection, 
this strategy aims to capture trends in both bull and bear markets while avoiding 
false signals during low-volatility periods. Designed for low trade frequency on daily timeframe.
"""

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
    
    # KAMA parameters
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    abs_sum = np.zeros_like(close)
    for i in range(10, len(close)):
        abs_sum[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.where(abs_sum[10:] == 0, 1, abs_sum[10:])
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss == 0, 0, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with close
    
    # ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - low[:-1])
    tr3 = np.abs(low[1:] - high[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR vs 50-period average (volatility regime filter)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / np.where(atr_ma == 0, 1, atr_ma)
    
    # Weekly trend filter (1w close > 20-period EMA)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 14, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(atr_ratio[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_ratio = atr_ratio[i]
        weekly_ema = ema_20_1w_aligned[i]
        
        # Volatility filter: only trade when volatility is elevated (avoid chop)
        if vol_ratio < 0.8 or vol_ratio > 2.5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA AND RSI > 50 AND weekly uptrend
            if price > kama_val and rsi_val > 50 and close_1w[i//7] > weekly_ema if i//7 < len(close_1w) else False:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA AND RSI < 50 AND weekly downtrend
            elif price < kama_val and rsi_val < 50 and close_1w[i//7] < weekly_ema if i//7 < len(close_1w) else False:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < KAMA OR RSI < 40
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > KAMA OR RSI > 60
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0