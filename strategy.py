#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Regime
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike confirmation, and choppiness regime filter. 
Only trade breakouts aligned with daily trend in non-choppy markets. Uses discrete position sizing (0.25) to minimize fee drag.
Target: 20-50 trades/year per symbol (~80-200 total over 4 years) to avoid fee drag.
Works in bull/bear via trend filter - only long in uptrend, short in downtrend.
Choppiness filter avoids whipsaws in ranging markets.
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
    
    # Get 1d data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla width
    rang = high_1d - low_1d
    
    # Resistance levels
    r3 = close_1d_prev + rang * 1.1 / 4
    r4 = close_1d_prev + rang * 1.1 / 2
    
    # Support levels
    s3 = close_1d_prev - rang * 1.1 / 4
    s4 = close_1d_prev - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detector (20-bar volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness regime filter (14-period)
    # Chop > 61.8 = ranging market (avoid), Chop < 38.2 = trending market (favor)
    hl_range = np.maximum(high, low) - np.minimum(high, low)  # True range approximation
    atr14 = pd.Series(hl_range).rolling(window=14, min_periods=14).mean().values
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14.sum() / (max_hh - min_ll)) / np.log10(14)
    # Handle division by zero and NaN
    chop = np.where((max_hh - min_ll) > 0, chop, 50.0)  # Neutral when no range
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)  # Align to 4h
    chop_filter = chop_aligned < 61.8  # Avoid choppy markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike in uptrend and non-choppy market
            if close[i] > r3_aligned[i] and volume_spike[i] and uptrend and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike in downtrend and non-choppy market
            elif close[i] < s3_aligned[i] and volume_spike[i] and downtrend and chop_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below R3 OR trend changes OR market becomes choppy
            if close[i] < r3_aligned[i] or not uptrend or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above S3 OR trend changes OR market becomes choppy
            if close[i] > s3_aligned[i] or not downtrend or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0