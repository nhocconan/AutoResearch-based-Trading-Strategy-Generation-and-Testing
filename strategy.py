#!/usr/bin/env python3
# 12h_1w_Keltner_Breakout_Trend_Filter
# Hypothesis: In trending markets, price breaking above/below Keltner Channel (2x ATR) on weekly timeframe
# indicates strong momentum. We use 12h close > weekly EMA50 for trend filter and volume confirmation
# to avoid false breakouts. Works in bull markets (follows uptrends) and bear markets (follows downtrends)
# by only trading in direction of 12h trend. Target: 15-30 trades/year to minimize fee drag.

name = "12h_1w_Keltner_Breakout_Trend_Filter"
timeframe = "12h"
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
    
    # Get weekly data for Keltner Channel and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly ATR for Keltner Channel (14-period)
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1w = tr.rolling(window=14, min_periods=14).mean().values
    upper_keltner = df_1w['close'] + 2 * atr_14_1w
    lower_keltner = df_1w['close'] - 2 * atr_14_1w
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA50 (50), ATR (14), volume MA (20)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 12h price vs weekly EMA50 for trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above upper Keltner + volume
            if uptrend and close[i] > upper_keltner_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below lower Keltner + volume
            elif downtrend and close[i] < lower_keltner_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters Keltner Channel
            if not uptrend or close[i] < upper_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters Keltner Channel
            if not downtrend or close[i] > lower_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals