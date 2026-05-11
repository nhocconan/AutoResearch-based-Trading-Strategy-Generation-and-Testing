#!/usr/bin/env python3
# 4h_KAMA_Trend_With_1d_RSI_Filter
# Hypothesis: In trending markets (ADX>25), KAMA adapts quickly to capture trends while filtering false signals with 1d RSI extremes. 
# Long when KAMA rises and price > KAMA with RSI < 30 (oversold in uptrend), short when KAMA falls and price < KAMA with RSI > 70 (overbought in downtrend).
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) by using RSI extremes as contrarian signals within the trend.

name = "4h_KAMA_Trend_With_1d_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for KAMA and 1d data for RSI and ADX
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4h KAMA (adaptive trend) ---
    close_4h = df_4h['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = np.sum(np.abs(np.diff(close_4h, axis=0)), axis=0) if len(close_4h) > 1 else np.zeros_like(close_4h)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    kama_4h = kama
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # --- 1d RSI for extreme conditions ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # --- 1d ADX for trend strength ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for KAMA (30), RSI (14), ADX (14+14)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_4h_aligned[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength filter: only trade in strong trends
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            if strong_trend:
                # Long: strong uptrend + price > KAMA + RSI oversold
                if close[i] > kama_4h_aligned[i] and rsi_1d_aligned[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # Short: strong downtrend + price < KAMA + RSI overbought
                elif close[i] < kama_4h_aligned[i] and rsi_1d_aligned[i] > 70:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price crosses below KAMA OR RSI overbought
                if close[i] < kama_4h_aligned[i] or rsi_1d_aligned[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above KAMA OR RSI oversold
                if close[i] > kama_4h_aligned[i] or rsi_1d_aligned[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals