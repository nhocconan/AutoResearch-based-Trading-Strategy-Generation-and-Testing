# -*- coding: utf-8 -*-
# -*- mode: python -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R combined with 1d EMA trend and volume spike
# Williams %R identifies overbought/oversold conditions (mean reversion)
# Trend filter (1d EMA50) ensures we trade with higher timeframe momentum
# Volume spike confirms conviction in the move
# Works in both bull/bear: In bull markets, buy dips in uptrend; in bear, sell rallies in downtrend
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1h data for Williams %R calculation (more responsive than 6h)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 14:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Williams %R (14-period) on 1h data
    highest_high = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6s timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1h, williams_r)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 6h volume for confirmation
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Williams %R conditions: oversold (< -80) or overbought (> -20)
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Volume filter: current volume above 20-period average
        volume_filter = volume[i] > vol_ma_6h[i] * 1.5
        
        # Long conditions: oversold + uptrend + volume spike
        long_condition = oversold and price_above_ema and volume_filter
        
        # Short conditions: overbought + downtrend + volume spike
        short_condition = overbought and price_below_ema and volume_filter
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: Williams %R returns to neutral territory
        elif position == 1 and williams_r_aligned[i] > -50:
            signals[i] = 0.0
            position = 0
        elif position == -1 and williams_r_aligned[i] < -50:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0