#!/usr/bin/env python3
"""
1h RSI(2) Pullback with 4h Trend Filter and Volume Spike
RSI(2) < 10 = oversold pullback in uptrend (long)
RSI(2) > 90 = overbought bounce in downtrend (short)
4h EMA50 defines trend: price > EMA50 = uptrend, price < EMA50 = downtrend
Volume must be > 1.5x 20-period average to confirm momentum
Target: 60-150 total trades over 4 years (15-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi2_pullback_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H TREND FILTER (HTF) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    ema4h_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema4h_50)  # already shifted
    
    # === RSI(2) CALCULATION ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[1] = gain[1]
    avg_loss[1] = loss[1]
    for i in range(2, n):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === VOLUME SPIKE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(ema4h_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend from 4h EMA50
        uptrend = close[i] > ema4h_50_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend turns down
            if rsi[i] > 70 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend turns up
            if rsi[i] < 30 or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i] * 1.5:
                signals[i] = 0.0
                continue
            
            # Entry logic based on 4h trend
            if uptrend:
                # In uptrend: long RSI(2) pullback (<10)
                if rsi[i] < 10:
                    position = 1
                    signals[i] = 0.20
            else:
                # In downtrend: short RSI(2) bounce (>90)
                if rsi[i] > 90:
                    position = -1
                    signals[i] = -0.20
    
    return signals