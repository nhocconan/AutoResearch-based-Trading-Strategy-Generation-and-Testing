#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_v2
# Hypothesis: 12h strategy using weekly Camarilla pivot levels with volume confirmation and ADX regime filter.
# Long: Price breaks above weekly R3 with volume > 1.3x 20-period average and ADX < 25 (range regime).
# Short: Price breaks below weekly S3 with volume > 1.3x 20-period average and ADX < 25 (range regime).
# Exit: Price returns to weekly pivot point (PP).
# Uses weekly Camarilla pivots from 1w timeframe as structure levels.
# Volume confirmation filters breakouts. ADX < 25 ensures we only trade in ranging markets.
# Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ADX for regime filter (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
    
    tr = np.zeros(n)
    for i in range(n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1]) if i > 0 else hl
        lc = abs(low[i] - close[i-1]) if i > 0 else hl
        tr[i] = max(hl, hc, lc)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for each 1w bar
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # Range = High - Low
    range_1w = high_1w - low_1w
    
    # Resistance levels
    r3 = pp + (range_1w * 3.0 / 8.0)
    
    # Support levels
    s3 = pp - (range_1w * 3.0 / 8.0)
    
    # Align Camarilla levels to 12h
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        # Regime filter: ADX < 25 (ranging market)
        ranging_regime = adx[i] < 25
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above weekly R3 with volume confirmation and ranging regime
            if (close[i] > r3_aligned[i] and volume_confirmed and ranging_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly S3 with volume confirmation and ranging regime
            elif (close[i] < s3_aligned[i] and volume_confirmed and ranging_regime):
                position = -1
                signals[i] = -0.25
    
    return signals