#!/usr/bin/env python3
# 12h_KAMA_RSI_Chop
# Hypothesis: On 12h timeframe, use KAMA trend direction combined with RSI extremes
# and Choppiness Index regime filter to capture mean reversion in choppy markets
# and trend continuation in trending markets. Designed for low trade frequency
# (10-25/year) to avoid fee drag, works in both bull and bear via regime adaptation.

name = "12h_KAMA_RSI_Chop"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Choppiness Index and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.absolute(high_1d[1:] - close_1d[:-1]), 
                     np.absolute(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[0], tr1])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Absolute price change
    abs_change = np.abs(close_1d[1:] - close_1d[:-1])
    abs_change = np.concatenate([[0], abs_change])
    sum_abs_change14 = pd.Series(abs_change).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(sum(abs_change)/atr) / log10(period)
    chop = 100 * np.log10(sum_abs_change14 / atr14) / np.log10(14)
    chop = np.where(atr14 > 0, chop, 50)  # default to middle when ATR is 0
    
    # Choppiness regime: > 61.8 = range (mean revert), < 38.2 = trending (trend follow)
    chop_range = chop > 61.8
    chop_trend = chop < 38.2
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h KAMA for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Efficiency Ratio
    change = np.abs(np.concatenate([[0], np.diff(close_12h)]))
    volatility = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0) if len(close_12h) > 1 else 0
    # Simplified ER calculation for array
    er = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        if i >= 10:
            change_10 = np.abs(close_12h[i] - close_12h[i-10])
            volatility_10 = np.sum(np.abs(np.diff(close_12h[i-10:i+1])))
            er[i] = change_10 / (volatility_10 + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align 1d indicators to 12h
    chop_range_aligned = align_htf_to_ltf(prices, df_1d, chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, df_1d, chop_trend.astype(float))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(chop_range_aligned[i]) or np.isnan(chop_trend_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(kama_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signals
            long_signal = False
            # In choppy market: mean reversion from RSI oversold
            if chop_range_aligned[i] > 0.5 and rsi_aligned[i] < 30:
                if close[i] > kama_aligned[i]:  # price above KAMA = bullish bias
                    long_signal = True
            # In trending market: trend continuation from RSI pullback
            elif chop_trend_aligned[i] > 0.5 and rsi_aligned[i] < 40:
                if close[i] > kama_aligned[i]:  # uptrend
                    long_signal = True
            
            # Short signals
            short_signal = False
            # In choppy market: mean reversion from RSI overbought
            if chop_range_aligned[i] > 0.5 and rsi_aligned[i] > 70:
                if close[i] < kama_aligned[i]:  # price below KAMA = bearish bias
                    short_signal = True
            # In trending market: trend continuation from RSI pullback
            elif chop_trend_aligned[i] > 0.5 and rsi_aligned[i] > 60:
                if close[i] < kama_aligned[i]:  # downtrend
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought or regime shift against position
            if (rsi_aligned[i] > 70 or 
                (chop_trend_aligned[i] > 0.5 and close[i] < kama_aligned[i]) or
                (chop_range_aligned[i] > 0.5 and rsi_aligned[i] > 50)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold or regime shift against position
            if (rsi_aligned[i] < 30 or 
                (chop_trend_aligned[i] > 0.5 and close[i] > kama_aligned[i]) or
                (chop_range_aligned[i] > 0.5 and rsi_aligned[i] < 50)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals