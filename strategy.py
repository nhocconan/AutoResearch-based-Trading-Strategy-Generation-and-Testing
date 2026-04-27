#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R reversal with 1-day volume confirmation and 1-day trend filter.
Trades reversals when Williams %R crosses below -80 (oversold) or above -20 (overbought)
with volume > 1.2x daily average and price on correct side of daily EMA(34).
Designed to work in both bull and bear markets by using daily trend as filter.
Target: 25-35 trades/year per symbol (100-140 total over 4 years) to minimize fee drag.
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
    
    # Get 12-hour data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12-hour Williams %R(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    
    # Align Williams %R to 12-hour timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get daily data for volume filter and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1-day EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Williams %R, volume MA, and daily EMA
    start_idx = max(14, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 12-hour price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1d = ema_34_1d_aligned[i]
        wr_now = williams_r_aligned[i]
        
        # Volume filter: volume > 1.2x 1-day average
        vol_filter = vol_now > 1.2 * vol_ma
        
        # Entry conditions: Williams %R reversal with volume and daily trend alignment
        if position == 0:
            # Long: Williams %R crosses below -80 (oversold) with volume + daily uptrend
            if wr_now <= -80 and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: Williams %R crosses above -20 (overbought) with volume + daily downtrend
            elif wr_now >= -20 and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R rises above -50 or daily trend turns down
            if wr_now >= -50 or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R falls below -50 or daily trend turns up
            if wr_now <= -50 or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsR_Reversal_1dVolume_1dTrend"
timeframe = "12h"
leverage = 1.0