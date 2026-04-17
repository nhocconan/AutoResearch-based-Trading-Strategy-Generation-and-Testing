#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h volume spike and 1d ADX trend filter.
Long when price breaks above R3 with volume > 2.0x 4h median volume AND 1d ADX > 25.
Short when price breaks below S3 with volume > 2.0x 4h median volume AND 1d ADX > 25.
Exit when price touches the opposite Camarilla level (S3 for long, R3 for short).
Uses 4h for volume confirmation and 1d for trend strength, 1h for entry timing.
Designed to capture strong intraday moves in trending markets while avoiding chop.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
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
    
    # Get 4h data for volume median (more stable than mean)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    vol_median_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20, center=False).median().values
    vol_median_20_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14) on 1d data
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx[0:14] = np.nan  # ensure warmup
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_median_20_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 4h median volume
        volume_confirmed = volume[i] > 2.0 * vol_median_20_aligned[i]
        
        # Trend filter: 1d ADX > 25 (strong trend)
        trend_strong = adx_aligned[i] > 25
        
        # Get the most recent completed 1d bar's OHLC for Camarilla
        high_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['high'].values)
        low_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['low'].values)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        
        period_high = high_1d_aligned[i]
        period_low = low_1d_aligned[i]
        period_close = close_1d_aligned[i]
        
        range_val = period_high - period_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels (R3/S3 for stronger breakouts)
        R3 = period_close + range_val * 1.1 / 4
        S3 = period_close - range_val * 1.1 / 4
        
        # Breakout conditions
        breakout_R3 = close[i] > R3
        breakout_S3 = close[i] < S3
        
        if position == 0:
            # Long: break above R3 with volume confirmation and strong trend
            if (breakout_R3 and volume_confirmed and trend_strong):
                signals[i] = 0.20
                position = 1
            # Short: break below S3 with volume confirmation and strong trend
            elif (breakout_S3 and volume_confirmed and trend_strong):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price touches S3
            if close[i] <= S3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price touches R3
            if close[i] >= R3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Volume_4h_ADX1d_Trend"
timeframe = "1h"
leverage = 1.0