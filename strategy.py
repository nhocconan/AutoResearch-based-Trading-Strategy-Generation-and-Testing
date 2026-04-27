#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams %R mean reversion with 1-day trend filter and volume confirmation.
Trades reversals when %R reaches extreme oversold/overbought levels, only in direction of daily trend,
with volume > 1.5x daily average to confirm momentum. Designed to work in both bull and bear
markets by using daily trend as filter and volume to avoid false signals in low-momentum periods.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
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
    
    # Get 6-hour data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    williams_r = williams_r.values
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get daily data for volume filter and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1-day EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Williams %R, volume MA, and daily EMA
    start_idx = max(14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 6h price, volume, and Williams %R
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        williams_r_now = williams_r_aligned[i]
        trend_1d = ema_50_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 1-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Williams %R extremes
        OVERSOLD = -80
        OVERBOUGHT = -20
        
        # Entry conditions: Williams %R reversal with volume and daily trend alignment
        if position == 0:
            # Long: %R oversold with volume + daily uptrend
            if williams_r_now <= OVERSOLD and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: %R overbought with volume + daily downtrend
            elif williams_r_now >= OVERBOUGHT and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: %R reaches center (-50) or daily trend turns down
            if williams_r_now >= -50 or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: %R reaches center (-50) or daily trend turns up
            if williams_r_now <= -50 or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsR_MeanReversion_1dVolume_1dTrend"
timeframe = "6h"
leverage = 1.0