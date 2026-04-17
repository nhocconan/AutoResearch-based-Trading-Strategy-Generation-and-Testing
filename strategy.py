#!/usr/bin/env python3
"""
Hypothesis: Daily pivot points act as strong support/resistance levels on lower timeframes.
When price approaches the daily pivot with increased volume and shows rejection (long lower shadow for longs,
long upper shadow for shorts), it often reverses. This strategy captures these reversals by entering 
long when price bounces off the pivot from below with bullish rejection and volume confirmation,
and short when price is rejected from the pivot from above with bearish rejection and volume confirmation.
Exits occur when price moves to the opposite pivot level (S1 for longs, R1 for shorts) or when
the rejection signal fails. Designed for 4h timeframe to work in both trending and ranging markets.
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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot and support/resistance levels
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    
    # Calculate support and resistance levels
    s1 = 2 * pivot - phigh
    r1 = 2 * pivot - plow
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all daily levels to 4h timeframe (waits for daily bar to close)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(r1_4h[i]) or
            np.isnan(ema_50_4h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        # Calculate candle body and shadows
        body = abs(price - close[i-1]) if i > 0 else 0
        lower_shadow = min(close[i-1], price) - low_price if i > 0 else 0
        upper_shadow = high_price - max(close[i-1], price) if i > 0 else 0
        
        if position == 0:
            # Long: price near S1 with bullish rejection (long lower shadow) and volume spike
            if (price <= s1_4h[i] * 1.005 and  # near S1 (within 0.5%)
                lower_shadow > body * 1.5 and   # long lower shadow
                vol > 1.5 * vol_ma):            # volume spike
                signals[i] = 0.25
                position = 1
            # Short: price near R1 with bearish rejection (long upper shadow) and volume spike
            elif (price >= r1_4h[i] * 0.995 and  # near R1 (within 0.5%)
                  upper_shadow > body * 1.5 and   # long upper shadow
                  vol > 1.5 * vol_ma):            # volume spike
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot or shows bearish rejection at resistance
            if (price >= pivot_4h[i] or  # reached pivot
                (price >= r1_4h[i] * 0.995 and upper_shadow > body * 1.5)):  # rejected at R1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or shows bullish rejection at support
            if (price <= pivot_4h[i] or  # reached pivot
                (price <= s1_4h[i] * 1.005 and lower_shadow > body * 1.5)):  # rejected at S1
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_Rejection_Volume"
timeframe = "4h"
leverage = 1.0