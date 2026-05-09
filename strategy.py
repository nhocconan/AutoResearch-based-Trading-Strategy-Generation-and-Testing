#!/usr/bin/env python3
# Hypothesis: 1h MACD histogram reversal with 4h EMA50 trend filter and volume spike
# Long when MACD histogram crosses above zero with 4h EMA50 uptrend and volume > 1.5x average
# Short when MACD histogram crosses below zero with 4h EMA50 downtrend and volume > 1.5x average
# Exit when MACD histogram crosses back to opposite side of zero
# Uses MACD for momentum reversal, EMA for trend, volume for conviction
# Designed to capture momentum shifts in both trending and ranging markets with controlled frequency
# Target: 60-150 total trades over 4 years (15-37/year) with size 0.20

name = "1h_MACD_ZeroCross_4hEMA50_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate MACD (12,26,9) on close prices
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and MACD calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(macd_hist[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: MACD histogram crosses above zero, 4h EMA50 uptrend, volume spike
            if (macd_hist[i] > 0 and macd_hist[i-1] <= 0 and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: MACD histogram crosses below zero, 4h EMA50 downtrend, volume spike
            elif (macd_hist[i] < 0 and macd_hist[i-1] >= 0 and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: MACD histogram crosses below zero
            if macd_hist[i] < 0 and macd_hist[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: MACD histogram crosses above zero
            if macd_hist[i] > 0 and macd_hist[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals