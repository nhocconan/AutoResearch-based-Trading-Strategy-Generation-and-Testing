#!/usr/bin/env python3
# 4h_RVOL_Trend_Breakout
# Strategy: 4h breakout of 20-period high/low with volume confirmation (RVOL > 1.5) and 1d EMA50 trend filter.
# Long when price breaks above 20-period high + RVOL > 1.5 + close > 1d EMA50.
# Short when price breaks below 20-period low + RVOL > 1.5 + close < 1d EMA50.
# Exit when price returns to 10-period SMA (mean reversion) or opposite breakout occurs.
# Designed for 4h timeframe to capture momentum with volume confirmation and trend filter.
# RVOL filters out low-volume breakouts, reducing false signals.
# EMA50 trend filter avoids counter-trend trades in strong trends.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "4h_RVOL_Trend_Breakout"
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
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 20-period high and low for breakout
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate 20-period average volume for RVOL
    avg_volume = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        avg_volume[i] = np.mean(volume[i - lookback + 1:i + 1])
    rvol = np.where(avg_volume > 0, volume / avg_volume, 0)
    
    # Calculate 10-period SMA for exit
    sma_10 = np.full(n, np.nan)
    for i in range(9, n):
        sma_10[i] = np.mean(close[i - 9:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(rvol[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(sma_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: breakout above 20-period high + RVOL > 1.5 + above 1d EMA50
            if close[i] > highest_high[i] and rvol[i] > 1.5 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: breakout below 20-period low + RVOL > 1.5 + below 1d EMA50
            elif close[i] < lowest_low[i] and rvol[i] > 1.5 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to 10-period SMA or opposite breakout
            if close[i] < sma_10[i] or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to 10-period SMA or opposite breakout
            if close[i] > sma_10[i] or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals