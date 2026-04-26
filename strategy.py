#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 12h with 1w EMA34 trend filter and volume confirmation. 
Targets 50-150 total trades over 4 years by requiring confluence of 1w trend, volume spike, and Camarilla breakout. 
Uses 1w trend to avoid counter-trend trades in bear markets, volume spike to confirm institutional interest, 
and Camarilla levels for precise entry/exit. Designed to work in both bull and bear markets via 1w trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d OHLC for Camarilla levels (using 1d as reference for pivot)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3, R4, S4
    # R4 = close + ((high - low) * 1.1 / 2)
    # R3 = close + ((high - low) * 1.1 / 4)
    # S3 = close - ((high - low) * 1.1 / 4)
    # S4 = close - ((high - low) * 1.1 / 2)
    camarilla_base = (high_1d + low_1d + close_1d * 2) / 4  # Typical price approximation
    rang = high_1d - low_1d
    r3 = camarilla_base + (rang * 1.1 / 4)
    s3 = camarilla_base - (rang * 1.1 / 4)
    r4 = camarilla_base + (rang * 1.1 / 2)
    s4 = camarilla_base - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike: volume > 2.0x 28-period median volume (≈14 days for 12h)
    volume_series = pd.Series(volume)
    vol_median_28 = volume_series.rolling(window=28, min_periods=28).median().values
    volume_spike = volume > (2.0 * vol_median_28)
    
    # Position size: 0.25 (25% of capital) to balance return and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1w EMA, 28 for volume median
    start_idx = max(34, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(vol_median_28[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above R3 with volume spike and 1w uptrend (close > EMA34_1w)
            long_entry = (high[i] > r3_aligned[i]) and vol_spike and (close_val > ema_34_val)
            # Short: price breaks below S3 with volume spike and 1w downtrend (close < EMA34_1w)
            short_entry = (low[i] < s3_aligned[i]) and vol_spike and (close_val < ema_34_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal, at R4 (take profit), or if price falls back below R3
            if close_val < ema_34_val or low[i] < r3_aligned[i] or high[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, at S4 (take profit), or if price rises back above S3
            if close_val > ema_34_val or high[i] > s3_aligned[i] or low[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0