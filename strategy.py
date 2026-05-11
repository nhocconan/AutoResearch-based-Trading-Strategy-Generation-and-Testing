#!/usr/bin/env python3
# 4h_1d_KAMA_Trend_RSI_Entry
# Hypothesis: Uses 1d KAMA to determine trend direction and 4h RSI for entry timing.
# In strong trends (ADX>25), price tends to pull back to the KAMA before continuing.
# Long when price is above 1d KAMA (uptrend) and RSI < 40 (pullback).
# Short when price is below 1d KAMA (downtrend) and RSI > 60 (pullback).
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Uses volume confirmation to avoid false signals.

name = "4h_1d_KAMA_Trend_RSI_Entry"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for KAMA and ADX
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d KAMA for trend ---
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close_1d - np.roll(close_1d, 10))
    change[0] = change[1] if len(change) > 1 else 0
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else 0
    # Simplified: use rolling sum of absolute changes
    vol_sum = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        vol_sum[i] = vol_sum[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    er = np.where(vol_sum > 0, change / vol_sum, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # --- 1d ADX for trend strength ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
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
    
    # --- 4h RSI for entry ---
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume confirmation ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for ADX (30) and RSI (14)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength filter: only trade in strong trends
        strong_trend = adx_aligned[i] > 25
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        if position == 0:
            if strong_trend and vol_filter:
                # Long: uptrend + price above KAMA + RSI oversold
                if close[i] > kama_1d_aligned[i] and rsi[i] < 40:
                    signals[i] = 0.25
                    position = 1
                # Short: downtrend + price below KAMA + RSI overbought
                elif close[i] < kama_1d_aligned[i] and rsi[i] > 60:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price crosses below KAMA OR RSI overbought
                if close[i] < kama_1d_aligned[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above KAMA OR RSI oversold
                if close[i] > kama_1d_aligned[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals