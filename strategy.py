#!/usr/bin/env python3
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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA(10,2,30) - adaptive moving average
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)
    er = np.zeros_like(close_1d)
    er[10:] = change[9:] / (volatility[9:] + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1w ATR(14) for volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above KAMA (bullish trend)
        # 2. RSI between 40 and 60 (neutral momentum, avoiding extremes)
        # 3. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        if (close[i] > kama_1d_aligned[i] and
            40 <= rsi_1d_aligned[i] <= 60 and
            atr_14_1w_aligned[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below KAMA (bearish trend)
        # 2. RSI between 40 and 60 (neutral momentum)
        # 3. Volatility filter: ATR > 0.3% of price
        elif (close[i] < kama_1d_aligned[i] and
              40 <= rsi_1d_aligned[i] <= 60 and
              atr_14_1w_aligned[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA10_RSI14_1w_ATR_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0