#!/usr/bin/env python3
name = "1d_Weekly_Trend_With_Daily_Volume_Filter"
timeframe = "1d"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily RSI(14) for mean reversion
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to daily timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Simple 20-period EMA for weekly trend
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily volume filter: > 1.5x 20-period average
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) with weekly uptrend and volume
            if (rsi_1d_aligned[i] < 30 and close[i] > ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) with weekly downtrend and volume
            elif (rsi_1d_aligned[i] > 70 and close[i] < ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 50 (mean reversion) or trend change
            if rsi_1d_aligned[i] > 50 or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 50 (mean reversion) or trend change
            if rsi_1d_aligned[i] < 50 or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily mean reversion with weekly trend filter and volume confirmation.
# In bull markets, buy oversold dips (RSI<30) in uptrend (price>weekly EMA).
# In bear markets, sell overbought rallies (RSI>70) in downtrend (price<weekly EMA).
# Volume filter ensures institutional participation. Works in both bull and bear markets.
# Target: 15-30 trades/year to minimize fee drag. Position size 0.25 limits risk.