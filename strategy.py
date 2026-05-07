#!/usr/bin/env python3
name = "4h_4H_TripleConfirmation_Strategy"
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
    
    # Get 1d data for multiple confirmations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d volume average
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 1d EMA20 (uptrend), volume above average, and price > open + 0.5 * ATR (momentum)
            if (close[i] > ema_20_1d_aligned[i] and 
                volume[i] > vol_avg_1d_aligned[i] and
                close[i] > prices['open'].iloc[i] + 0.5 * atr_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA20 (downtrend), volume above average, and price < open - 0.5 * ATR (momentum)
            elif (close[i] < ema_20_1d_aligned[i] and 
                  volume[i] > vol_avg_1d_aligned[i] and
                  close[i] < prices['open'].iloc[i] - 0.5 * atr_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA20 (trend change) or volatility drops
            if close[i] < ema_20_1d_aligned[i] or volume[i] < vol_avg_1d_aligned[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA20 (trend change) or volatility drops
            if close[i] > ema_20_1d_aligned[i] or volume[i] < vol_avg_1d_aligned[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals