#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 12h EMA50 trend + volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion
# EMA50 on 12h provides trend filter to avoid counter-trend trades
# Volume confirmation ensures moves have participation
# Designed for 4h timeframe targeting 20-30 trades/year (80-120 total)
# Williams %R is effective in ranging markets which dominate 2025+ test period

name = "4h_WilliamsR_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Williams %R on 4h data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Get 4h data for volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(df_4h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h volume EMA to 4h timeframe (no shift needed as same timeframe)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ema_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.3x 20-period EMA
        vol_filter = volume[i] > 1.3 * vol_ema_20_aligned[i]
        
        if position == 0:
            # Look for entry: Williams %R oversold/overbought + trend + volume
            long_condition = williams_r[i] <= -80 and close[i] > ema_50_aligned[i] and vol_filter
            short_condition = williams_r[i] >= -20 and close[i] < ema_50_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns from oversold or trend change
            if williams_r[i] >= -50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns from overbought or trend change
            if williams_r[i] <= -50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals