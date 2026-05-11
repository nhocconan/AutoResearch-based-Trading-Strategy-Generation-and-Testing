#!/usr/bin/env python3
# 1d_1w_KAMA_Trend_RSI_Entry
# Hypothesis: Uses 1d KAMA direction as primary trend filter (adaptive moving average
# that reduces noise during chop) combined with 1d RSI for mean-reversion entries
# within the trend. In strong trends (adx>25 on 1w), we look for RSI extremes
# to enter in the direction of the 1w trend. Works in bull markets (buy dips in
# uptrend) and bear markets (sell rallies in downtrend). Low-frequency signals
# to avoid fee drag on 1d timeframe.

name = "1d_1w_KAMA_Trend_RSI_Entry"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w data for trend and 1d data for signals
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w KAMA for trend direction ---
    close_1w = df_1w['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(close_1w - np.roll(close_1w, 10))
    volatility = np.sum(np.abs(np.diff(close_1w, prepend=close_1w[0])), axis=0)  # placeholder - will compute properly
    # Proper volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i >= 10:
            volatility[i] = np.sum(np.abs(np.diff(close_1w[i-10:i+1])))
        else:
            volatility[i] = 1e-10
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # --- 1w ADX for trend strength ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
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
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # --- 1d RSI for entry signals ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1w indicators (30) and RSI (14)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_1w_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength filter: only trade in strong trends
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:
            if strong_trend:
                # Determine trend direction from 1w KAMA
                uptrend = close[i] > kama_1w_aligned[i]
                downtrend = close[i] < kama_1w_aligned[i]
                
                # Long: uptrend + RSI oversold
                if uptrend and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                # Short: downtrend + RSI overbought
                elif downtrend and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: RSI returns to neutral OR price crosses below KAMA
                if rsi[i] > 50 or close[i] < kama_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI returns to neutral OR price crosses above KAMA
                if rsi[i] < 50 or close[i] > kama_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals