#!/usr/bin/env python3
name = "1d_KAMA_RSI_ChopFilter_v1"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA on daily close
    close_series = pd.Series(close)
    change = abs(close_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change.rolling(window=10, min_periods=10).sum() / volatility.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_series = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean()
    ema_1w = ema_1w_series.values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14) - regime filter
    atr1 = pd.Series(np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])).rolling(window=14, min_periods=14).sum()
    atr1 = np.concatenate([[np.nan], atr1.values])
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1 / (hh - ll)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA (trend up) AND RSI < 40 (pullback) AND chop > 61.8 (range)
            if close[i] > kama[i] and rsi[i] < 40 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA (trend down) AND RSI > 60 (pullback) AND chop > 61.8 (range)
            elif close[i] < kama[i] and rsi[i] > 60 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < KAMA OR RSI > 70 (overbought)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > KAMA OR RSI < 30 (oversold)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals