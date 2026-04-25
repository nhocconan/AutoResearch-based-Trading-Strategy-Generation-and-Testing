#!/usr/bin/env python3
"""
1h Williams %R Mean Reversion + 4h EMA Trend + Volume Spike
Hypothesis: Williams %R extremes on 1h indicate short-term overbought/oversold conditions.
In the direction of 4h EMA50 trend with volume confirmation, these provide high-probability mean reversion entries.
Works in bull markets via pullback longs in uptrends and bear markets via rally shorts in downtrends.
Designed for 1h timeframe targeting 15-35 trades/year with strict entry filters to avoid fee drag.
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
    
    # Get 4h data for EMA50 trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Williams %R(14) on 1h
    if len(close) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        williams_r[highest_high == lowest_low] = -50.0
    else:
        williams_r = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Williams %R, EMA50_4h, and volume MA to propagate
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema50_4h = ema_50_4h_aligned[i]
        wr = williams_r[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.8 * 20-period average
        volume_spike = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend (price > 4h EMA50) AND volume spike
            long_condition = (wr < -80) and (curr_close > ema50_4h) and volume_spike
            # Short: Williams %R overbought (> -20) AND downtrend (price < 4h EMA50) AND volume spike
            short_condition = (wr > -20) and (curr_close < ema50_4h) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: Williams %R returns above -50 (mean reversion complete) or adverse move
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Williams %R returns below -50 (mean reversion complete) or adverse move
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_WilliamsR_MeanReversion_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0