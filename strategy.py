#!/usr/bin/env python3
# 6h_RSI_EMA_Trend_1dFilter
# Strategy: Trade RSI extremes with EMA trend filter on 1d timeframe
# Long when RSI(14) < 30 and price > 1d EMA(50)
# Short when RSI(14) > 70 and price < 1d EMA(50)
# Exit when RSI returns to neutral zone (40-60)
# Uses mean reversion in ranging markets with trend filter to avoid counter-trend trades
# Designed for 6h timeframe with selective entries to minimize trade frequency

name = "6h_RSI_EMA_Trend_1dFilter"
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
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[1:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    gain_smooth = wilders_smooth(gain, 14)
    loss_smooth = wilders_smooth(loss, 14)
    
    rs = np.where(loss_smooth != 0, gain_smooth / loss_smooth, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI oversold and above 1d EMA50 (uptrend filter)
            if rsi[i] < 30 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought and below 1d EMA50 (downtrend filter)
            elif rsi[i] > 70 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral zone
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral zone
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals