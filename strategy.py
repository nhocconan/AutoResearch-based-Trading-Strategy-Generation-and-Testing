#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Reversal_v1"
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
    
    # KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        vol = np.abs(np.diff(close, prepend=close[0]))
        er = np.zeros_like(close)
        for i in range(1, len(close)):
            er[i] = change[i] / (vol[i] + 1e-10) if vol[i] > 0 else 0
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(alpha=1/length, adjust=False).mean().values
        max_high = pd.Series(high).rolling(length, min_periods=length).max().values
        min_low = pd.Series(low).rolling(length, min_periods=length).min().values
        chop = 100 * np.log10(atr * length / (max_high - min_low + 1e-10)) / np.log10(length)
        return chop
    
    # Calculate indicators
    kama_val = kama(close, 10, 2, 30)
    rsi_val = rsi(close, 14)
    chop_val = choppiness_index(high, low, close, 14)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Volume filter: current volume > 1.5 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(chop_val[i]) or np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition
        vol_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price > KAMA, RSI < 30 (oversold), Chop > 61.8 (ranging), weekly uptrend
            if (close[i] > kama_val[i] and rsi_val[i] < 30 and 
                chop_val[i] > 61.8 and close[i] > sma50_1w_aligned[i] and vol_condition):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA, RSI > 70 (overbought), Chop > 61.8 (ranging), weekly downtrend
            elif (close[i] < kama_val[i] and rsi_val[i] > 70 and 
                  chop_val[i] > 61.8 and close[i] < sma50_1w_aligned[i] and vol_condition):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA or RSI > 70 (overbought) or Chop < 38.2 (trending)
            if (close[i] < kama_val[i] or rsi_val[i] > 70 or chop_val[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA or RSI < 30 (oversold) or Chop < 38.2 (trending)
            if (close[i] > kama_val[i] or rsi_val[i] < 30 or chop_val[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals