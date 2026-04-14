#!/usr/bin/env python3
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
    
    # Load weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Supertrend (ATR=10, mult=3)
    def calculate_supertrend(high, low, close, period=10, multiplier=3):
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.full(len(high), np.nan)
        if len(high) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        hl2 = (high + low) / 2
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        
        supertrend = np.full(len(high), np.nan)
        direction = np.full(len(high), 1)  # 1 for uptrend, -1 for downtrend
        
        for i in range(period, len(high)):
            if close[i] > upper[i-1]:
                direction[i] = 1
            elif close[i] < lower[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1:
                supertrend[i] = max(lower[i], supertrend[i-1])
            else:
                supertrend[i] = min(upper[i], supertrend[i-1])
        
        return supertrend, direction
    
    supertrend_1w, trend_1w = calculate_supertrend(high_1w, low_1w, close_1w, 10, 3)
    
    # Align weekly Supertrend to daily timeframe
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Calculate daily ATR for volatility filter
    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(len(close), np.nan)
    if len(close) >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, len(close)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate daily EMA for trend filter
    ema_period = 50
    ema = np.full(len(close), np.nan)
    if len(close) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema[ema_period-1] = np.mean(close[:ema_period])
        for i in range(ema_period, len(close)):
            ema[i] = (close[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_1w_aligned[i]) or 
            np.isnan(trend_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(ema[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 1% of price)
        if atr[i] < 0.01 * close[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above EMA50 AND weekly Supertrend uptrend
            if (close[i] > ema[i] and 
                trend_1w_aligned[i] == 1):
                position = 1
                signals[i] = position_size
            # Short: Price below EMA50 AND weekly Supertrend downtrend
            elif (close[i] < ema[i] and 
                  trend_1w_aligned[i] == -1):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls below EMA50 OR weekly trend turns down
            if (close[i] < ema[i] or 
                trend_1w_aligned[i] == -1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises above EMA50 OR weekly trend turns up
            if (close[i] > ema[i] or 
                trend_1w_aligned[i] == 1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklySupertrend_EMA_Filter"
timeframe = "1d"
leverage = 1.0