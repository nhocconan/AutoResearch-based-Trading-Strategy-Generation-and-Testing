#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day RSI (14) mean reversion in oversold/overbought zones, confirmed by 4h EMA (50) trend.
# Enters long when 1d RSI < 30 (oversold) and price > 4h EMA50 (uptrend), exits when RSI > 50.
# Enters short when 1d RSI > 70 (overbought) and price < 4h EMA50 (downtrend), exits when RSI < 50.
# Uses 1d RSI for mean reversion signal and 4h EMA50 for trend filter to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_RSI_MeanReversion_EMA50_Trend"
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
    
    # Calculate 1-day RSI (14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Calculate 4h EMA (50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close']
    ema_50_4h = close_4h.ewm(span=50, adjust=False).mean()
    ema_50_4h_values = ema_50_4h.values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: 1d RSI < 30 (oversold) and price > 4h EMA50 (uptrend)
            if rsi_1d_aligned[i] < 30 and close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: 1d RSI > 70 (overbought) and price < 4h EMA50 (downtrend)
            elif rsi_1d_aligned[i] > 70 and close[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 1d RSI > 50 (mean reversion complete)
            if rsi_1d_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 1d RSI < 50 (mean reversion complete)
            if rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals