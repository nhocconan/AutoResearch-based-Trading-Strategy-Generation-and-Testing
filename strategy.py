#!/usr/bin/env python3
name = "1d_200day_MA_With_RSI_And_Price_Position"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for 200-day MA (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 200-day EMA (more responsive than SMA for trend)
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get weekly data for trend confirmation (stronger filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 50-week EMA for weekly trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI on daily (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe (daily)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Price position relative to 200-day EMA (how far above/below as percentage)
    # Normalize to [-1, 1] range where 0 = at EMA, positive = above, negative = below
    price_to_ema_ratio = (close_1d - ema200_1d) / (ema200_1d + 1e-10)
    # Clip extreme values to avoid instability
    price_to_ema_ratio = np.clip(price_to_ema_ratio, -0.5, 0.5)
    price_pos_aligned = align_htf_to_ltf(prices, df_1d, price_to_ema_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for 200-day EMA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(price_pos_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. Price above 200-day EMA (uptrend)
            # 2. Weekly trend also up (50-week EMA)
            # 3. RSI not overbought (< 70) 
            # 4. Price not too far above EMA (avoid chasing)
            if (close_1d[i] > ema200_1d_aligned[i] and 
                ema50_1w_aligned[i] > 0 and  # Weekly EMA has meaningful value
                rsi_aligned[i] < 70 and 
                price_pos_aligned[i] > -0.1 and  # Not significantly below EMA
                price_pos_aligned[i] < 0.2):     # Not significantly above EMA
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Price below 200-day EMA (downtrend)
            # 2. Weekly trend also down
            # 3. RSI not oversold (> 30)
            # 4. Price not too far below EMA (avoid catching falling knife)
            elif (close_1d[i] < ema200_1d_aligned[i] and 
                  ema50_1w_aligned[i] < 0 and  # Weekly EMA has meaningful negative value
                  rsi_aligned[i] > 30 and 
                  price_pos_aligned[i] < 0.1 and   # Not significantly above EMA
                  price_pos_aligned[i] > -0.2):    # Not significantly below EMA
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 200-day EMA OR RSI overbought
            if close_1d[i] < ema200_1d_aligned[i] or rsi_aligned[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 200-day EMA OR RSI oversold
            if close_1d[i] > ema200_1d_aligned[i] or rsi_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals