#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams %R mean reversion with 1-day trend filter and volume confirmation.
Buys when W%R < -80 (oversold) in 1-day uptrend, sells when W%R > -20 (overbought) in 1-day downtrend.
Uses volume spike to confirm mean reversion bounce. Designed for range-bound markets with occasional trends.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
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
    
    # Get 1-day data for trend filter and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1-day Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    willr = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    
    # Calculate 1-day EMA(50) for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1-day indicators to 6-hour timeframe
    willr_aligned = align_htf_to_ltf(prices, df_1d, willr)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 6-day data for volume filter (using 6h close as proxy for recent volume context)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    vol_6h = df_6h['volume'].values
    vol_ma_20_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Williams %R, EMA, and volume MA
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(willr_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        willr_val = willr_aligned[i]
        trend_1d = ema_50_1d_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_6h_aligned[i]
        
        # Volume filter: volume > 2.0x 6h average (strict to reduce trades)
        vol_filter = vol_now > 2.0 * vol_ma
        
        # Entry conditions: Williams %R mean reversion with trend and volume
        if position == 0:
            # Long: W%R < -80 (oversold) + 1-day uptrend + volume spike
            if willr_val < -80 and close[i] > trend_1d and vol_filter:
                signals[i] = size
                position = 1
            # Short: W%R > -20 (overbought) + 1-day downtrend + volume spike
            elif willr_val > -20 and close[i] < trend_1d and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: W%R > -50 (return to midpoint) or trend breakdown
            if willr_val > -50 or close[i] < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: W%R < -50 (return to midpoint) or trend reversal
            if willr_val < -50 or close[i] > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsR_MeanReversion_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0