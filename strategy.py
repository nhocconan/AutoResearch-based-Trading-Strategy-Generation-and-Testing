#!/usr/bin/env python3
# Hypothesis: 6h Exponential Moving Average Crossover with 1d MACD Trend Filter and Volume Confirmation
# Long when 6h EMA20 crosses above EMA50, 1d MACD histogram is positive, and volume > 2x average
# Short when 6h EMA20 crosses below EMA50, 1d MACD histogram is negative, and volume > 2x average
# Uses EMA crossover for momentum, 1d MACD for trend direction, volume for conviction
# Designed to capture medium-term trends with controlled frequency in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_EMA20_50_Crossover_1dMACD_Volume"
timeframe = "6h"
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
    
    # Calculate 1d MACD components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Align 1d MACD histogram to 6h timeframe
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    # Calculate 6h EMA20 and EMA50
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50 calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(macd_hist_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: EMA20 crosses above EMA50, MACD histogram positive, volume spike
            if (ema20[i] > ema50[i] and ema20[i-1] <= ema50[i-1] and
                macd_hist_aligned[i] > 0 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: EMA20 crosses below EMA50, MACD histogram negative, volume spike
            elif (ema20[i] < ema50[i] and ema20[i-1] >= ema50[i-1] and
                  macd_hist_aligned[i] < 0 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: EMA20 crosses below EMA50
            if ema20[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EMA20 crosses above EMA50
            if ema20[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals