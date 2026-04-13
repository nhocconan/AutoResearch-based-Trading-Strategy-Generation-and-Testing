# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1-day Williams %R extremes with volume confirmation.
In bear markets (like 2025), extreme oversold conditions on daily timeframe
often precede short-term bounces. We go long when daily %R < -80 (oversold)
and volume confirms buying interest. Exit when %R > -20 (overbought) or trend
fails. This captures mean reversion within larger downtrends while avoiding
whipsaws via volume filter.
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
    
    # Get daily data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    
    # Handle first value
    highest_high[0] = high_1d[0]
    lowest_low[0] = low_1d[0]
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1, hl_range)
    
    williams_r = -100 * (highest_high - close_1d) / hl_range
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily volume SMA20 for volume confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    # Start after enough data for indicators
    start_idx = max(50, 20)  # Williams %R needs 14, EMA50 needs 50, vol SMA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-period average
        volume_confirmed = volume[i] > vol_sma_20_aligned[i]
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Trend filter: price above EMA50 (avoid buying in strong downtrends)
        uptrend_filter = close[i] > ema_50_1d_aligned[i]
        
        # Entry conditions: oversold + volume + mild uptrend filter
        long_entry = oversold and volume_confirmed and uptrend_filter
        
        # Exit conditions: overbought or trend breakdown
        exit_long = position == 1 and (overbought or not uptrend_filter)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif exit_long:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else 0.0
    
    return signals

name = "4h_1d_williams_r_volume_filter_v1"
timeframe = "4h"
leverage = 1.0