#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyKAMA_Pullback"
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
    
    # Weekly KAMA for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # KAMA components
    change = np.abs(np.diff(df_1w['close'].values, prepend=df_1w['close'].values[0]))
    volatility = np.abs(np.diff(df_1w['close'].values))
    er = change / (volatility + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(df_1w['close'].values)
    kama[0] = df_1w['close'].values[0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (df_1w['close'].values[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Daily ATR for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily RSI for pullback entries
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.2 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(kama_aligned[i]) or np.isnan(atr[i]) or \
           np.isnan(rsi[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above weekly KAMA (trend), RSI pullback (oversold), with volume
            if (price > kama_aligned[i] and 
                rsi[i] < 35 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price below weekly KAMA (trend), RSI pullback (overbought), with volume
            elif (price < kama_aligned[i] and 
                  rsi[i] > 65 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price crosses below weekly KAMA or RSI overbought
            if (price < kama_aligned[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly KAMA or RSI oversold
            if (price > kama_aligned[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals