#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_v1
KAMA(14,10) for trend direction + RSI(14) + Choppiness Index(14) regime filter.
Long when KAMA rising, RSI>50, CHOP<40 (trending). Short when KAMA falling, RSI<50, CHOP<40.
Uses 1w EMA50 as higher timeframe trend filter.
Target: 50-100 total trades over 4 years (12-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === KAMA(14,10) ===
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, 14))
    change[0:14] = np.nan
    volatility = np.abs(np.diff(close, prepend=np.nan))
    volatility_sum = pd.Series(volatility).rolling(window=14, min_periods=14).sum().values
    er = change / volatility_sum
    er[0:14] = np.nan
    # Smoothing constants
    sc = (er * (2/(10+1) - 2/(30+1)) + 2/(30+1))**2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[14] = close[14]  # seed
    for i in range(15, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index(14) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    
    # === 1w EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA rising, RSI>50, CHOP<40 (trending), price above 1w EMA50
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                chop[i] < 40 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA falling, RSI<50, CHOP<40 (trending), price below 1w EMA50
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  chop[i] < 40 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: KAMA falling OR RSI<40 OR CHOP>61.8 (choppy)
            if (kama[i] < kama[i-1] or 
                rsi[i] < 40 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA rising OR RSI>60 OR CHOP>61.8 (choppy)
            if (kama[i] > kama[i-1] or 
                rsi[i] > 60 or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0