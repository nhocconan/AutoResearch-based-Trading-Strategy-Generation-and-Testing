#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Donchian breakout with 1-day ATR-based volatility filter and 1-week EMA trend filter.
Donchian channels capture breakouts from volatility contractions; ATR filter ensures trades occur during elevated volatility.
Weekly EMA provides trend direction filter to avoid counter-trend trades. Designed for 20-30 trades/year to minimize fee drag.
Works in bull markets (buy upper band breaks in uptrend) and bear markets (sell lower band breaks in downtrend).
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
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(10) on 1d data
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    atr_1d = np.full(len(close_1d), np.nan)
    if len(tr_1d) >= 10:
        atr_1d[9] = np.mean(tr_1d[:10])
        for i in range(10, len(tr_1d)):
            atr_1d[i] = (tr_1d[i] + 9 * atr_1d[i-1]) / 10
    
    # Align ATR to 4h timeframe
    atr_1d_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 1w data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2/51) + (ema_50_1w[i-1] * 49/51)
    
    # Align 1w EMA to 4h timeframe
    ema_50_1w_4h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need EMA, ATR, Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1w_4h[i]) or np.isnan(atr_1d_4h[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 1.5 * 10-period average ATR
        if i >= 50:
            atr_ma = np.nanmean(atr_1d_4h[i-10:i])
            vol_filter = not np.isnan(atr_ma) and atr_1d_4h[i] > 1.5 * atr_ma
        else:
            vol_filter = False
        
        # Trend filter: price above/below 1w EMA50
        trend_up = close[i] > ema_50_1w_4h[i]
        trend_down = close[i] < ema_50_1w_4h[i]
        
        if position == 0:
            # Long entry: close above Donchian upper band with volatility and uptrend
            if (close[i] > donchian_high[i] and 
                vol_filter and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below Donchian lower band with volatility and downtrend
            elif (close[i] < donchian_low[i] and 
                  vol_filter and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below Donchian lower band
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian upper band
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRVolFilter_1wEMA50"
timeframe = "4h"
leverage = 1.0