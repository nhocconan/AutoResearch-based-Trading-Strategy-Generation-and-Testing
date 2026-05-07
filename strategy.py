#!/usr/bin/env python3
name = "6h_RSI_20_EMA50_Crossover_1dTrend_Volume"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 60-period RSI(20) for mean reversion signal
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initialize first average
    if len(close) > 20:
        avg_gain[20] = np.mean(gain[1:21])
        avg_loss[20] = np.mean(loss[1:21])
    
    # Wilder smoothing
    for i in range(21, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 19 + gain[i-1]) / 20
        avg_loss[i] = (avg_loss[i-1] * 19 + loss[i-1]) / 20
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 60-period volume filter: > 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 21)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 20 (oversold) AND price above 1d EMA50 (uptrend filter) with volume
            if (rsi[i] < 20 and close[i] > ema_50_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 80 (overbought) AND price below 1d EMA50 (downtrend filter) with volume
            elif (rsi[i] > 80 and close[i] < ema_50_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 60 (overbought threshold) or price below EMA50 (trend change)
            if rsi[i] > 60 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 40 (oversold threshold) or price above EMA50 (trend change)
            if rsi[i] < 40 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s timeframe with RSI(20) mean reversion + 1d EMA50 trend filter + volume confirmation.
# RSI < 20 for long, RSI > 80 for short - captures extreme mean reversion moments.
# 1d EMA50 ensures we only trade with the higher timeframe trend (long in uptrend, short in downtrend).
# Volume filter ensures sufficient participation.
# Works in bull markets (longs when RSI oversold in uptrend) and bear markets (shorts when RSI overbought in downtrend).
# Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag. Position size 0.25 limits risk.