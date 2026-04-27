#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R reversal with 1-day volume confirmation and 1-day trend filter.
Trades reversals when Williams %R reaches oversold/overbought levels with volume spike and daily trend alignment.
Designed to work in both bull and bear markets by using daily trend as filter and volume to confirm reversal strength.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drift.
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
    
    # Get 4-hour data for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4-hour Williams %R (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    wr = (highest_high - close_4h) / (highest_high - lowest_low) * -100
    
    # Align Williams %R to 4-hour timeframe
    wr_aligned = align_htf_to_ltf(prices, df_4h, wr)
    
    # Get daily data for volume filter and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1-day EMA(25) for trend
    close_1d = df_1d['close'].values
    ema_25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_25_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Williams %R, volume MA, and daily EMA
    start_idx = max(14, 20, 25)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(wr_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_25_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 4-hour price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1d = ema_25_1d_aligned[i]
        wr_now = wr_aligned[i]
        
        # Volume filter: volume > 1.3x 1-day average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Entry conditions: Williams %R reversal with volume and daily trend alignment
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume + daily uptrend
            if wr_now < -80 and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought (> -20) with volume + daily downtrend
            elif wr_now > -20 and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R reaches overbought (> -20) or daily trend turns down
            if wr_now > -20 or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R reaches oversold (< -80) or daily trend turns up
            if wr_now < -80 or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsR_Reversal_1dVolume_1dTrend"
timeframe = "4h"
leverage = 1.0