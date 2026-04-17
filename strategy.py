#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h/1d Camarilla pivot breakout + volume confirmation.
Long when price breaks above R3 with volume > 1.5x 20-period average.
Short when price breaks below S3 with volume > 1.5x 20-period average.
Use 1d ADX > 20 to filter for trending markets and avoid ranging whipsaws.
Session filter: 08-20 UTC to reduce noise.
Position sizing: 0.20 for entries.
Target: 60-150 total trades over 4 years (15-37/year).
Camarilla pivots provide precise support/resistance levels that work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Typical price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels: R3 = close + (high - low) * 1.1/2
    # Support levels: S3 = close - (high - low) * 1.1/2
    r3 = close_1d + range_1d * 1.1 / 2
    s3 = close_1d - range_1d * 1.1 / 2
    
    # Calculate 1d ADX (14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all to 1h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume filter and ADX > 20
            if (close[i] > r3_aligned[i] and 
                volume_filter[i] and 
                adx_aligned[i] > 20):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with volume filter and ADX > 20
            elif (close[i] < s3_aligned[i] and 
                  volume_filter[i] and 
                  adx_aligned[i] > 20):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S3 or ADX weakens
            if (close[i] < s3_aligned[i] or 
                adx_aligned[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises above R3 or ADX weakens
            if (close[i] > r3_aligned[i] or 
                adx_aligned[i] < 15):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_1dCamarilla_R3S3_Volume_ADX_Session"
timeframe = "1h"
leverage = 1.0