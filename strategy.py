#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter
Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter and chop regime filter.
Designed for 1d timeframe targeting 30-100 trades over 4 years (7-25/year).
Uses weekly EMA50 for trend direction and daily chop filter (BW percentile) to avoid whipsaws.
Volume confirmation on breakout. Works in bull/bear markets: In trending regimes (price > weekly EMA50 for longs, < for shorts),
breakouts at R1/S1 with volume spike capture momentum. Chop filter prevents entries in sideways markets.
Exit on trend reversal or range re-entry (close beyond R3/S3).
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels from previous daily bar (completed)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    r3 = prev_close + (rng * 1.1 / 4)
    s3 = prev_close - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 1d (already aligned as 1d->1d, but keep for consistency)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Bollinger Band Width for chop regime (20, 2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Chop filter: BW percentile over 50d, < 0.3 = trending (favor breakouts), > 0.7 = choppy (avoid)
    # Calculate percentile rank manually to avoid look-ahead
    bb_width_percentile = np.full(n, np.nan)
    lookback = 50
    for i in range(lookback, n):
        window = bb_width[i-lookback:i+1]
        rank = (window <= bb_width[i]).sum() / len(window)
        bb_width_percentile[i] = rank
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25  # 25% position size
    
    # Warmup: need 1w EMA50, 1d shift, BB width percentile, vol avg
    start_idx = max(50, 30, 20, 20)  # 1w EMA50, 1d shift, BB, vol
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_50_1w_aligned[i]
        chop_filter = bb_width_percentile[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with weekly trend alignment,
            # volume spike, and NOT in choppy regime (chop filter < 0.7)
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            chop_filter < 0.7)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             chop_filter < 0.7)
            
            if long_condition:
                signals[i] = base_size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -base_size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price re-enters Camarilla range (below S1) OR loses weekly trend alignment
            # Stronger exit: break S3 or close below weekly EMA50
            if close_val < s3_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = base_size
        elif position == -1:
            # Exit short: price re-enters Camarilla range (above R1) OR loses weekly trend alignment
            # Stronger exit: break R3 or close above weekly EMA50
            if close_val > r3_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter"
timeframe = "1d"
leverage = 1.0