#!/usr/bin/env python3
# 4h_KAMA_Trend_Filter_With_RSI_Pullback_v2
# Hypothesis: Combines Kaufman Adaptive Moving Average (KAMA) trend direction with RSI pullback entries for timely reversals.
# Uses 12h trend filter to avoid counter-trend trades, improving performance in both bull and bear markets.
# Volume confirmation ensures institutional participation. Designed for low trade frequency (~20-30/year) to minimize fee drag.

name = "4h_KAMA_Trend_Filter_With_RSI_Pullback_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate KAMA on 12h close
    close_12h = df_12h['close'].values
    delta = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    er = np.abs(np.diff(close_12h, n=10)) / (np.sum(delta[np.arange(10, len(close_12h))[:, None] == np.arange(len(delta))[:, None, None]], axis=1))
    er = np.concatenate([np.full(10, np.nan), er])
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Align KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Warmup for volume MA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 12h KAMA
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price above KAMA (uptrend) + RSI pullback from oversold + volume
            if uptrend and rsi_aligned[i] < 35 and rsi_aligned[i] > np.nanmin(rsi_aligned[max(0, i-5):i]) and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA (downtrend) + RSI pullback from overbought + volume
            elif downtrend and rsi_aligned[i] > 65 and rsi_aligned[i] < np.nanmax(rsi_aligned[max(0, i-5):i]) and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend turns down or RSI overbought
            if not uptrend or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend turns up or RSI oversold
            if not downtrend or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals