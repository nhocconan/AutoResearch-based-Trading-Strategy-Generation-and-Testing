#!/usr/bin/env python3
# 4h_KAMA_Trend_With_Price_Action_Filter
# Hypothesis: KAMA adapts to market efficiency, reducing lag in trends and noise in ranges.
# In trending markets, price stays above/below KAMA with momentum. We add price action
# confirmation: price must close beyond KAMA + ATR multiplier to avoid whipsaws.
# Works in bull markets (follows uptrends) and bear markets (follows downtrends) by
# only trading in direction of KAMA trend. Uses volume confirmation to avoid false signals.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_KAMA_Trend_With_Price_Action_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[-1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[i] - close[i-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum |close - close[-1]| over 10
    # Avoid division by zero
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i-10]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # Get 1d data for trend filter (slower, more reliable)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for volatility and price action filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full_like(close, np.nan)
    for i in range(14, len(tr)):
        if not np.isnan(tr[i-14:i]).any():
            atr[i] = np.mean(tr[i-14:i])
    
    # Volume confirmation (20-period MA on 4h = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), ATR (14), EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters: both KAMA and 1d EMA50 must agree
        kama_uptrend = close[i] > kama[i]
        kama_downtrend = close[i] < kama[i]
        ema_uptrend = close[i] > ema_50_1d_aligned[i]
        ema_downtrend = close[i] < ema_50_1d_aligned[i]
        
        uptrend = kama_uptrend and ema_uptrend
        downtrend = kama_downtrend and ema_downtrend
        
        # Price action filter: price must be beyond KAMA by 0.5*ATR
        pa_long = close[i] > kama[i] + 0.5 * atr[i]
        pa_short = close[i] < kama[i] - 0.5 * atr[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price action + volume
            if uptrend and pa_long and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price action + volume
            elif downtrend and pa_short and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price action reverses
            if not uptrend or not pa_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price action reverses
            if not downtrend or not pa_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals