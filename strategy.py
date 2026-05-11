#!/usr/bin/env python3
name = "4h_RSI_Momentum_Filter"
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
    
    # 1h data for RSI momentum and trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    
    # Calculate 14-period RSI on 1h
    delta = pd.Series(close_1h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1h = (100 - (100 / (1 + rs))).values
    
    # Smooth RSI with 3-period SMA to reduce noise
    rsi_smooth_1h = pd.Series(rsi_1h).rolling(window=3, min_periods=3).mean().values
    
    # Align to 4h
    rsi_aligned = align_htf_to_ltf(prices, df_1h, rsi_smooth_1h)
    
    # 4h EMA21 for trend filter
    ema21_4h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4h volume average for volume confirmation
    vol_avg_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema21_4h[i]) or 
            np.isnan(vol_avg_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI > 55 and rising, price above EMA21, volume above average
            if (rsi_aligned[i] > 55 and 
                rsi_aligned[i] > rsi_aligned[i-1] and
                close[i] > ema21_4h[i] and
                volume[i] > vol_avg_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 45 and falling, price below EMA21, volume above average
            elif (rsi_aligned[i] < 45 and 
                  rsi_aligned[i] < rsi_aligned[i-1] and
                  close[i] < ema21_4h[i] and
                  volume[i] > vol_avg_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI < 50 or price crosses below EMA21
            if (rsi_aligned[i] < 50 or 
                close[i] < ema21_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI > 50 or price crosses above EMA21
            if (rsi_aligned[i] > 50 or 
                close[i] > ema21_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals