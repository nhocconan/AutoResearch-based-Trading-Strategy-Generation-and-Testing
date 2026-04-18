#!/usr/bin/env python3
"""
1h RSI(2) Pullback with 4h Trend and Volume Confirmation
Long: RSI(2) < 10 + price > 4h EMA50 + volume > 1.5x 20-bar avg
Short: RSI(2) > 90 + price < 4h EMA50 + volume > 1.5x 20-bar avg
Exit: RSI(2) crosses above 50 (long) or below 50 (short)
Designed for 1h timeframe to capture mean reversion within higher timeframe trend.
Target: 80-120 total trades over 4 years (20-30/year)
"""

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
    
    # RSI(2) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[1] = gain[1]
    avg_loss[1] = loss[1]
    
    for i in range(2, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need RSI and EMA calculations
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: RSI(2) < 10 + price > 4h EMA50 + volume filter
            if (rsi[i] < 10 and 
                price > ema_50_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 + price < 4h EMA50 + volume filter
            elif (rsi[i] > 90 and 
                  price < ema_50_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI(2) crosses above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI(2) crosses below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_Pullback_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0