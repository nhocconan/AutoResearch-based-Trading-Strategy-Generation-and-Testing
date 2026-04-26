#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_1wPivotFilter
Hypothesis: 6h Donchian(20) breakouts filtered by 1d EMA50 trend and weekly Camarilla S3/R3 levels to avoid counter-trend trades. Works in bull/bear markets by only taking breakouts in the direction of the 1d trend, while weekly pivot levels act as stronger support/resistance to filter false breakouts. Volume confirmation (>1.5x 20-bar MA) reduces whipsaws. Target: 12-37 trades/year (50-150 total over 4 years) with discrete position sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous week's OHLC for Camarilla S3/R3 (stronger support/resistance)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for weekly timeframe: S3/R3
    rng_1w = high_1w - low_1w
    camarilla_s3_1w = close_1w - (rng_1w * 1.1 / 4)  # S3 level
    camarilla_r3_1w = close_1w + (rng_1w * 1.1 / 4)  # R3 level
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w)
    camarilla_r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # 6h Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for Donchian/vol, 50 for 1d EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1w_aligned[i]) or 
            np.isnan(camarilla_r3_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        ema_50_val = ema_50_1d_aligned[i]
        camarilla_s3_val = camarilla_s3_1w_aligned[i]
        camarilla_r3_val = camarilla_r3_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Donchian breakout conditions
        breakout_up = close_val > highest_high_val
        breakout_down = close_val < lowest_low_val
        
        # Entry conditions: Donchian breakout in trend direction, not near weekly S3/R3 (avoid false breakouts), with volume spike
        long_entry = breakout_up and bullish_1d and (close_val > camarilla_s3_val) and vol_spike
        short_entry = breakout_down and bearish_1d and (close_val < camarilla_r3_val) and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Donchian mid-line or trend change
            donchian_mid = (highest_high_val + lowest_low_val) / 2
            if close_val < donchian_mid or not bullish_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on Donchian mid-line or trend change
            donchian_mid = (highest_high_val + lowest_low_val) / 2
            if close_val > donchian_mid or not bearish_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_1wPivotFilter"
timeframe = "6h"
leverage = 1.0