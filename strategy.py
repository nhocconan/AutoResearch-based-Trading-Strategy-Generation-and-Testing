#!/usr/bin/env python3
"""
1h Bollinger Band Reversal with 4h Trend and Volume Spike Filter.
Long when price touches lower BB + 4h uptrend + volume spike.
Short when price touches upper BB + 4h downtrend + volume spike.
Exit when price crosses back inside Bollinger Bands or trend changes.
Designed for low frequency (15-30 trades/year) to minimize fee drag.
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
    
    # Session filter: 08:00-20:00 UTC (precomputed)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend
    close_4h = df_4h['close'].values
    ema_20_4h = np.full_like(close_4h, np.nan, dtype=np.float64)
    alpha = 2.0 / (20 + 1)
    for i in range(len(close_4h)):
        if i < 20:
            if i == 0:
                ema_20_4h[i] = close_4h[i]
            else:
                ema_20_4h[i] = (close_4h[i] * alpha) + (ema_20_4h[i-1] * (1 - alpha))
        else:
            ema_20_4h[i] = (close_4h[i] * alpha) + (ema_20_4h[i-1] * (1 - alpha))
    
    # Align 4h EMA20 to 1h with proper delay
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Bollinger Bands on 1h
    bb_length = 20
    bb_std = 2.0
    sma = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if i >= bb_length - 1:
            sma[i] = np.mean(close[i-bb_length+1:i+1])
            std[i] = np.std(close[i-bb_length+1:i+1])
    
    upper = sma + (bb_std * std)
    lower = sma - (bb_std * std)
    
    # Volume filter: volume > 1.5x average (to avoid false signals)
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need Bollinger Bands (20) + volume MA (20)
    start_idx = max(bb_length, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(sma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        vol_now = volume[i]
        
        # Bollinger Bands
        upper_band = upper[i]
        lower_band = lower[i]
        
        # 4h EMA20 trend
        ema_20 = ema_20_4h_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price at or below lower BB + 4h uptrend + volume spike
            if price_now <= lower_band and price_now > ema_20 and vol_filter:
                signals[i] = size
                position = 1
            # Short: price at or above upper BB + 4h downtrend + volume spike
            elif price_now >= upper_band and price_now < ema_20 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back inside Bollinger Bands or 4h trend turns down
            if price_now >= sma[i] or price_now < ema_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back inside Bollinger Bands or 4h trend turns up
            if price_now <= sma[i] or price_now > ema_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_BollingerReversal_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0