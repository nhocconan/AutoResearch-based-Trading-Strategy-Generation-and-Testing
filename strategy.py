#!/usr/bin/env python3
name = "6h_RSI_Divergence_Volume_Trend"
timeframe = "6h"
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
    
    # Calculate RSI(14) on 6h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish RSI divergence + uptrend + volume confirmation
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (i >= 3 and 
                close[i] < close[i-2] and  # Lower low in price
                rsi[i] > rsi[i-2] and     # Higher low in RSI
                close[i] > ema_50_1d_aligned[i] and  # Uptrend filter
                volume_ratio[i] > 1.5):   # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence + downtrend + volume confirmation
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (i >= 3 and 
                  close[i] > close[i-2] and  # Higher high in price
                  rsi[i] < rsi[i-2] and     # Lower high in RSI
                  close[i] < ema_50_1d_aligned[i] and  # Downtrend filter
                  volume_ratio[i] > 1.5):   # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish RSI divergence or trend change
            if (i >= 3 and 
                close[i] > close[i-2] and  # Higher high in price
                rsi[i] < rsi[i-2]):       # Lower high in RSI (bearish divergence)
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_50_1d_aligned[i]:  # Trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish RSI divergence or trend change
            if (i >= 3 and 
                close[i] < close[i-2] and  # Lower low in price
                rsi[i] > rsi[i-2]):       # Higher low in RSI (bullish divergence)
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_50_1d_aligned[i]:  # Trend change
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals