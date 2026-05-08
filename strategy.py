#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Direction_RSI_Chop_Filter_v1"
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
    
    # 1. KAMA direction (1d)
    er = np.abs(close - np.roll(close, 9)) / np.sum(np.abs(np.diff(close, n=1)), axis=0, dtype=np.float64)
    er_series = pd.Series(er, index=prices.index)
    er_ma = er_series.rolling(window=10, min_periods=10).mean().values
    sc = (er_ma * 0.6645 + 0.0645) ** 2
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 2. RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 3. Choppiness Index (14)
    atr = np.abs(np.subtract.outer(high, low))
    tr1 = np.abs(np.subtract(high, np.roll(close, 1)))
    tr2 = np.abs(np.subtract(low, np.roll(close, 1)))
    tr = np.maximum.reduce([atr, tr1, tr2])
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    
    # 4. Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA, RSI > 50, chop < 61.8 (trending), price > weekly EMA50
            long_cond = (close[i] > kama[i]) and (rsi[i] > 50) and (chop[i] < 61.8) and (close[i] > ema_50_1w_aligned[i])
            # Short: price < KAMA, RSI < 50, chop < 61.8 (trending), price < weekly EMA50
            short_cond = (close[i] < kama[i]) and (rsi[i] < 50) and (chop[i] < 61.8) and (close[i] < ema_50_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA OR chop > 61.8 (choppy)
            if close[i] < kama[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA OR chop > 61.8 (choppy)
            if close[i] > kama[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals