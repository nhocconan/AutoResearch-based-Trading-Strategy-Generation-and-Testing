#!/usr/bin/env python3
# 1d_1w_momentum_v1
# Hypothesis: 1-day momentum with 1-week trend filter and volume confirmation.
# Long when daily price crosses above 20-day EMA, weekly trend is up (price > 10-week SMA), and volume > 1.5x average.
# Short when daily price crosses below 20-day EMA, weekly trend is down (price < 10-week SMA), and volume > 1.5x average.
# Exit when price returns to 10-day EMA or volume drops below 1.2x average.
# Works in bull via momentum continuation, in bear via shorting weakness during downtrends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_momentum_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 20-day EMA for daily momentum
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 10-day EMA for exit
    ema_10 = close_s.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 10-week SMA for weekly trend
    close_1w = pd.Series(df_1w['close'].values)
    sma_10w = close_1w.rolling(window=10, min_periods=10).mean().values
    sma_10w_aligned = align_htf_to_ltf(prices, df_1w, sma_10w)
    
    # Calculate volume moving average (20-day)
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema_20[i]) or np.isnan(ema_10[i]) or 
            np.isnan(sma_10w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 10-day EMA or volume drops below 1.2x average
            if price <= ema_10[i] or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 10-day EMA or volume drops below 1.2x average
            if price >= ema_10[i] or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price crosses above 20-day EMA, weekly trend up, volume expansion
            if price > ema_20[i] and price > sma_10w_aligned[i] and vol_ratio > 1.5:
                # Confirm crossover: previous price was at or below EMA
                if i > 0 and close[i-1] <= ema_20[i-1]:
                    position = 1
                    signals[i] = 0.25
            # Enter short: price crosses below 20-day EMA, weekly trend down, volume expansion
            elif price < ema_20[i] and price < sma_10w_aligned[i] and vol_ratio > 1.5:
                # Confirm crossover: previous price was at or above EMA
                if i > 0 and close[i-1] >= ema_20[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals