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
    
    # Load 1d data for price channel and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d data for volatility filter
    if len(high_1d) < 14:
        return np.zeros(n)
    
    tr = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i],
                   abs(high_1d[i] - high_1d[i-1]),
                   abs(low_1d[i] - low_1d[i-1]))
    
    atr_1d = np.full_like(high_1d, np.nan)
    if len(high_1d) >= 14:
        atr_1d[13] = np.mean(tr[1:14])
        for i in range(14, len(high_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 1d timeframe (it's already on 1d)
    # Now calculate 10-day and 20-day high/low channels
    high_20 = np.full_like(high_1d, np.nan)
    low_20 = np.full_like(low_1d, np.nan)
    high_10 = np.full_like(high_1d, np.nan)
    low_10 = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            high_20[i] = np.max(high_1d[i-19:i+1])
            low_20[i] = np.min(low_1d[i-19:i+1])
        if i >= 9:
            high_10[i] = np.max(high_1d[i-9:i+1])
            low_10[i] = np.min(low_1d[i-9:i+1])
    
    # Align channels to daily timeframe (already aligned)
    # For 1d timeframe, we use the values directly
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(n):
        # Skip if insufficient data for channels
        if i < 19:  # Need 20 bars for 20-day channel
            signals[i] = 0.0
            continue
            
        # Check volatility filter: only trade when ATR is above its 20-period average
        if i >= 19:
            atr_ma_20 = np.mean(atr_1d[i-19:i+1]) if not np.any(np.isnan(atr_1d[i-19:i+1])) else np.nan
            if np.isnan(atr_ma_20) or atr_1d[i] < 0.8 * atr_ma_20:  # Low volatility filter
                signals[i] = 0.0
                continue
        
        if position == 0:
            # Long: price breaks above 20-day high AND above 10-day high (strong breakout)
            if (close[i] > high_20[i] and close[i] > high_10[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 20-day low AND below 10-day low
            elif (close[i] < low_20[i] and close[i] < low_10[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below 10-day low (trailing exit)
            if close[i] < low_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises above 10-day high
            if close[i] > high_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_DualChannel_Breakout_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0