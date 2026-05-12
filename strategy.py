#!/usr/bin/env python3
"""
12h_RSI_MeanReversion_4hTrend_VolumeFilter
Hypothesis: Mean reversion on 12h RSI (oversold/overbought) with 4h trend filter and volume confirmation works in both bull and bear markets. In uptrends, buy RSI < 30; in downtrends, short RSI > 70. Volume > 1.5x average filters low-probability signals. Trend filter uses 4h EMA50 to avoid counter-trend trades. Targets 20-50 trades over 4 years to minimize fee drag.
"""

name = "12h_RSI_MeanReversion_4hTrend_VolumeFilter"
timeframe = "12h"
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
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume filter: >1.5x 20-period average (12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 12h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI < 30 (oversold) + 4h EMA50 uptrend + volume filter
            if (rsi[i] < 30 and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought) + 4h EMA50 downtrend + volume filter
            elif (rsi[i] > 70 and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 (mean reversion complete) or trend change
            if rsi[i] > 50 or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 (mean reversion complete) or trend change
            if rsi[i] < 50 or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals