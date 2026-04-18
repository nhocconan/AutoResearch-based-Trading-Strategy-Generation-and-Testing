#!/usr/bin/env python3
"""
6h_RSI_Range_Bound_With_Volume_Confirmation
Hypothesis: In range-bound markets, RSI extremes with volume confirmation provide mean-reversion opportunities. 
Go long when RSI < 30 and volume > 1.5x average, short when RSI > 70 and volume > 1.5x average.
Uses 12h trend filter to avoid trading against the trend. Designed for sideways markets common in 2025.
Target: 15-25 trades/year with position size 0.25.
"""

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
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use pandas for EMA calculation with proper handling
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length (first 14 values are invalid)
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA (34-period) for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above EMA = uptrend, below EMA = downtrend
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        if position == 0:
            # Long entry: RSI oversold with volume confirmation in uptrend or ranging market
            if rsi[i] < 30 and vol_confirmed and (uptrend or not downtrend):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought with volume confirmation in downtrend or ranging market
            elif rsi[i] > 70 and vol_confirmed and (downtrend or not uptrend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI returns to neutral zone (50)
            if rsi[i] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral zone (50)
            if rsi[i] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI_Range_Bound_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0