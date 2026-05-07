#!/usr/bin/env python3
name = "4h_RVOL_Breakout_v1"
timeframe = "4h"
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d data for RVOL calculation (current volume vs 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    # Calculate RVOL (relative volume) for 4h bar
    # RVOL = current 4h volume / average 4h volume over last 20 periods
    vol_ma20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = volume / vol_ma20_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma20_aligned[i]) or 
            np.isnan(rvol[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 1d EMA50 (uptrend), RVOL > 1.5 (volume spike), price > open (bullish candle)
            if (close[i] > ema_50_1d_aligned[i] and 
                rvol[i] > 1.5 and 
                close[i] > prices['open'].iloc[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA50 (downtrend), RVOL > 1.5 (volume spike), price < open (bearish candle)
            elif (close[i] < ema_50_1d_aligned[i] and 
                  rvol[i] > 1.5 and 
                  close[i] < prices['open'].iloc[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend change)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend change)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals