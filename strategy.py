#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1d RSI mean reversion with 12h trend filter.
# In bear markets, RSI extremes on daily timeframe often precede reversals.
# Uses 12h EMA(50) to filter for trend direction, taking mean reversion trades
# only in the direction of the higher timeframe trend to avoid counter-trend losses.
# Designed for low trade frequency (10-20/year) with clear entry/exit rules.

name = "6h_RSI_MeanReversion_12hTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily timeframe
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    rsi_1d[:14] = np.nan
    
    # Align RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 6h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long when RSI < 30 (oversold) and price above 12h EMA50 (uptrend)
            if rsi_aligned[i] < 30 and close[i] > ema_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short when RSI > 70 (overbought) and price below 12h EMA50 (downtrend)
            elif rsi_aligned[i] > 70 and close[i] < ema_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when RSI > 50 (mean reversion complete) or trend turns down
            if rsi_aligned[i] > 50 or close[i] < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when RSI < 50 (mean reversion complete) or trend turns up
            if rsi_aligned[i] < 50 or close[i] > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals