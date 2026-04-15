#!/usr/bin/env python3
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
    
    # 1d KAMA for trend direction (ER=10)
    price_diff = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(price_diff.reshape(-1, 10), axis=1) if len(price_diff) >= 10 else np.full_like(close, 1e-10)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # 1w Choppiness Index (CHOP)
    df_1w = get_htf_data(prices, '1w')
    tr_1w = np.maximum(df_1w['high'].values - df_1w['low'].values,
                       np.maximum(np.abs(df_1w['high'].values - np.concatenate([[df_1w['close'][0]], df_1w['close'][:-1]])),
                                  np.abs(df_1w['low'].values - np.concatenate([[df_1w['close'][0]], df_1w['close'][:-1]]))))
    atr_1w_sum = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_1w_sum / np.log10(14) / (highest_high - lowest_low))
    
    # Align 1w CHOP to daily
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Volume filter: current > 1.5x 20-day median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price > KAMA, RSI < 40 (pullback in uptrend), CHOP > 61.8 (range), volume confirmation
        if (close[i] > kama[i] and rsi[i] < 40 and chop_aligned[i] > 61.8 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price < KAMA, RSI > 60 (pullback in downtrend), CHOP > 61.8 (range), volume confirmation
        elif (close[i] < kama[i] and rsi[i] > 60 and chop_aligned[i] > 61.8 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: when trend weakens (price crosses KAMA) or chop drops (trending market)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < kama[i]) or
               (signals[i-1] == -0.25 and close[i] > kama[i]) or
               chop_aligned[i] < 38.2)):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_KAMA_RSI_CHOP_Volume"
timeframe = "1d"
leverage = 1.0