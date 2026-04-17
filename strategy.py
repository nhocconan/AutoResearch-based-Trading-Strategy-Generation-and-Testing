#!/usr/bin/env python3
"""
1h_4H1D_Trend_Filter_Momentum
Strategy: 1h momentum with 4h/1d trend filter
Long: RSI(14) > 55 + price > EMA(20) + 4h close > EMA(50) + 1d close > EMA(100)
Short: RSI(14) < 45 + price < EMA(20) + 4h close < EMA(50) + 1d close < EMA(100)
Exit: RSI returns to neutral zone (45-55)
Position size: 0.20
Session filter: 08-20 UTC
Designed to capture momentum bursts only in strong multi-timeframe trends.
"""

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
    
    # Calculate EMAs
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA50 (trend filter)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA100 (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_100_1d_aligned[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: RSI > 55 + price > EMA20 + 4h trend up + 1d trend up
            if (rsi[i] > 55 and close[i] > ema_20[i] and 
                ema_50_4h_aligned[i] > ema_50[i] and 
                ema_100_1d_aligned[i] > ema_100[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI < 45 + price < EMA20 + 4h trend down + 1d trend down
            elif (rsi[i] < 45 and close[i] < ema_20[i] and 
                  ema_50_4h_aligned[i] < ema_50[i] and 
                  ema_100_1d_aligned[i] < ema_100[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI drops below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI rises above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4H1D_Trend_Filter_Momentum"
timeframe = "1h"
leverage = 1.0