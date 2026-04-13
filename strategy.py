#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h KAMA trend with 1d RSI extremes and chop regime filter.
    # Long when 12h KAMA rising AND 1d RSI < 30 (oversold) AND chop > 61.8 (range regime).
    # Short when 12h KAMA falling AND 1d RSI > 70 (overbought) AND chop > 61.8.
    # Uses discrete position sizing (0.25) to target 50-150 trades over 4 years.
    # Works in bull/bear via chop filter avoiding trend-following false signals in ranging markets.
    # KAMA adapts to market noise, reducing whipsaw in choppy conditions.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for RSI and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h KAMA (adaptive moving average)
    # Efficiency Ratio: |close - close[10]| / sum(|close - close[1]| over 10 periods)
    change = np.abs(np.subtract(close[9:], np.roll(close, 10)[9:]))  # |close - close[10]|
    volatility = np.abs(np.subtract(close[1:], close[:-1]))  # |close - close[1]|
    er = np.zeros_like(change)
    for i in range(len(change)):
        if np.sum(volatility[i:i+10]) > 0:
            er[i] = change[i] / np.sum(volatility[i:i+10])
        else:
            er[i] = 0
    er = np.pad(er, (9, 0), mode='constant', constant_values=np.nan)  # align with close
    
    # Smoothing constants: fastest SC=2/(2+1)=0.667, slowest SC=2/(30+1)=0.0645
    sc = er * (0.667 - 0.0645) + 0.0645
    alpha = sc * sc  # ER squared for smoothing
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        if not np.isnan(alpha[i]):
            kama[i] = kama[i-1] + alpha[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align with close_1d
    
    # Calculate 1d chop regime: ATR(14) / (highest high - lowest low over 14) * 100 * log10(sqrt(14))/log10(10)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(atr_14 * np.sqrt(14) / range_14) / np.log10(10), 50)
    
    # Align 1d indicators to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA on daily, aligned to 12h
    
    # Calculate KAMA slope (rising/falling)
    kama_slope = np.diff(kama_aligned, prepend=np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(kama_slope[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # KAMA trend: rising or falling
        kama_rising = kama_slope[i] > 0
        kama_falling = kama_slope[i] < 0
        
        # RSI extremes
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Entry conditions
        if kama_rising and rsi_oversold and chop_filter and position != 1:
            position = 1
            signals[i] = position_size
        elif kama_falling and rsi_overbought and chop_filter and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions: opposite signal or chop regime breaks down
        elif position == 1 and (kama_falling or not chop_filter or rsi_aligned[i] > 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (kama_rising or not chop_filter or rsi_aligned[i] < 50):
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0