#!/usr/bin/env python3
"""
Hypothesis: 1-hour momentum with 4-hour trend filter and 1-day volume confirmation.
Long when price > 4h EMA50 (uptrend), RSI(14) > 55, and volume > 1.5x 24-period average.
Short when price < 4h EMA50 (downtrend), RSI(14) < 45, and volume > 1.5x 24-period average.
Exit when RSI crosses back to 50 (mean reversion within trend).
Uses 4h for trend direction, 1d for volume regime filter, 1h for entry timing.
Designed for low trade frequency (~20-40/year) to avoid fee drag in 1h timeframe.
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
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 24:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_avg_24 = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    vol_avg_24_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_24)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg_24_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend (price > 4h EMA50), bullish momentum (RSI > 55), volume confirmation
            if (close[i] > ema_50_4h_aligned[i] and 
                rsi[i] > 55 and 
                volume[i] > 1.5 * vol_avg_24_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Downtrend (price < 4h EMA50), bearish momentum (RSI < 45), volume confirmation
            elif (close[i] < ema_50_4h_aligned[i] and 
                  rsi[i] < 45 and 
                  volume[i] > 1.5 * vol_avg_24_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit when RSI crosses back to 50 (mean reversion within trend)
            exit_signal = False
            if position == 1 and rsi[i] < 50:
                exit_signal = True
            elif position == -1 and rsi[i] > 50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_4hEMA50_RSI_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0