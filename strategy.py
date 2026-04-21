#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA direction + RSI + chop filter with 1w trend filter.
Longs when KAMA is rising, RSI>50, and chop<61.8 (trending) with 1w EMA50 uptrend.
Shorts when KAMA is falling, RSI<50, chop<61.8, and 1w EMA50 downtrend.
Uses ATR-based stoploss. Designed for 15-30 trades/year to minimize fee drag while capturing
trend continuation in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for KAMA, RSI, chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    er[1:] = change[1:] / (np.sum(volatility[np.arange(1, len(close_1d))[:, None] <= np.arange(1, len(close_1d))[None, :]], axis=1) + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr_1d = []
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr_1d, axis=1) / (max_hh - min_ll + 1e-10)) / np.log10(14)
    chop = np.where((max_hh - min_ll) == 0, 50, chop)
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR for stoploss (20-period)
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: KAMA rising, RSI>50, trending, weekly uptrend
            if (kama_val > kama_aligned[i-1] and 
                rsi_val > 50 and 
                chop_val < 61.8 and 
                ema_50_1w_val > ema_50_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI<50, trending, weekly downtrend
            elif (kama_val < kama_aligned[i-1] and 
                  rsi_val < 50 and 
                  chop_val < 61.8 and 
                  ema_50_1w_val < ema_50_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: ATR-based stoploss (2x ATR from KAMA) or KAMA reversal
            exit_signal = False
            
            # ATR-based stoploss
            if position == 1:
                if price_close < kama_val - 2.0 * atr_val:
                    exit_signal = True
            elif position == -1:
                if price_close > kama_val + 2.0 * atr_val:
                    exit_signal = True
            
            # KAMA reversal exit
            if position == 1 and kama_val < kama_aligned[i-1]:
                exit_signal = True
            elif position == -1 and kama_val > kama_aligned[i-1]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_1wEMA50_Trend_ATR2x"
timeframe = "1d"
leverage = 1.0