#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Trend_Volume_Exhaustion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h price action
    highest_high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    range_size = highest_high_4h - lowest_low_4h
    upper_zone_4h = lowest_low_4h + 0.8 * range_size  # top 20%
    lower_zone_4h = lowest_low_4h + 0.2 * range_size  # bottom 20%
    
    # 12h trend filter (exponential moving average)
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma
    
    # Exhaustion signal: price at extreme of range with volume but against trend
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(ema_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long exhaustion: price at lower extreme + volume spike + but 12h trend is up (mean reversion)
            if (price <= lower_zone_4h[i] and    # at bottom 20% of range
                volume_filter[i] and             # volume confirmation
                price > ema_12h_aligned[i]):     # but 12h trend is up (fade the move)
                signals[i] = 0.25
                position = 1
                continue
            
            # Short exhaustion: price at upper extreme + volume spike + but 12h trend is down
            elif (price >= upper_zone_4h[i] and  # at top 20% of range
                  volume_filter[i] and           # volume confirmation
                  price < ema_12h_aligned[i]):   # but 12h trend is down (fade the move)
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price moves back toward middle or 12h trend accelerates up
            if (price >= (highest_high_4h[i] + lowest_low_4h[i]) / 2 or  # back to mid-range
                price > ema_12h_aligned[i] * 1.02):                       # trend acceleration
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price moves back toward middle or 12h trend accelerates down
            if (price <= (highest_high_4h[i] + lowest_low_4h[i]) / 2 or  # back to mid-range
                price < ema_12h_aligned[i] * 0.98):                       # trend acceleration
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals