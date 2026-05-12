#!/usr/bin/env python3
# 6h_RSI_Trend_With_200EMA_Filter
# Hypothesis: RSI(14) crossing above 50 with price above 200EMA signals momentum in trending markets.
# The 200EMA filter ensures we only trade in the direction of the long-term trend,
# reducing false signals during ranging periods. Works in both bull and bear markets
# by adapting to the prevailing trend direction. Target: 50-150 trades over 4 years.

name = "6h_RSI_Trend_With_200EMA_Filter"
timeframe = "6h"
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
    
    # === Daily Data for 200EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema_200_1d = pd.Series(daily_close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_6h = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === RSI (14-period) on 6h chart ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure EMA200 and RSI ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_200_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI > 50 (bullish momentum) + price above daily EMA200 (uptrend)
            if rsi[i] > 50 and close[i] > ema_200_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 50 (bearish momentum) + price below daily EMA200 (downtrend)
            elif rsi[i] < 50 and close[i] < ema_200_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI falls below 50 (momentum shifts)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI rises above 50 (momentum shifts)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals