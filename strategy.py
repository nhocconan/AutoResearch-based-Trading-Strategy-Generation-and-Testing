#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_RSI_Trend_v1"
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
    
    # Daily trend: EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Weekly trend: EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 12h KAMA for trend direction
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    vol = np.sum(np.abs(np.diff(close, prepend=close[0])))
    er = np.where(vol != 0, change / vol, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 12h RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h ATR(14) for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    # Volume filter: volume > 1.1x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.1 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or \
           np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above KAMA, above daily/weekly EMA, RSI > 50, with volume
            if (price > kama[i] and 
                price > ema34_1d_aligned[i] and 
                price > ema34_1w_aligned[i] and 
                rsi[i] > 50 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, below daily/weekly EMA, RSI < 50, with volume
            elif (price < kama[i] and 
                  price < ema34_1d_aligned[i] and 
                  price < ema34_1w_aligned[i] and 
                  rsi[i] < 50 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI < 40
            if (price < kama[i] or 
                rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI > 60
            if (price > kama[i] or 
                rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals