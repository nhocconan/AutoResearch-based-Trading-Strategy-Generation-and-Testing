#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_VolumeSpike_1dTrend_v6
Hypothesis: Camarilla pivot reversals with daily trend filter and volume spike capture mean reversion in both bull and bear markets.
Daily EMA filter ensures trades align with intermediate trend. Volume spike confirms participation. Designed for 20-40 trades/year to minimize fee drag.
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
    
    # Get daily data for Camarilla pivots and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # True range for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = close_1d[0]
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - prev_close), np.abs(low_1d - prev_close)))
    
    # Camarilla levels (using typical formula)
    R4 = close_1d + (tr * 1.1 / 2)
    R3 = close_1d + (tr * 1.1 / 4)
    R2 = close_1d + (tr * 1.1 / 6)
    R1 = close_1d + (tr * 1.1 / 12)
    S1 = close_1d - (tr * 1.1 / 12)
    S2 = close_1d - (tr * 1.1 / 6)
    S3 = close_1d - (tr * 1.1 / 4)
    S4 = close_1d - (tr * 1.1 / 2)
    
    # Use R3/S3 as entry levels, R4/S4 as stop levels
    # Shift by 1 to use previous day's levels only
    R3_1d = np.roll(R3, 1)
    S3_1d = np.roll(S3, 1)
    R4_1d = np.roll(R4, 1)
    S4_1d = np.roll(S4, 1)
    R3_1d[0] = np.nan
    S3_1d[0] = np.nan
    R4_1d[0] = np.nan
    S4_1d[0] = np.nan
    
    # Align to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3_1d)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4_1d)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Daily EMA for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(R4_4h[i]) or np.isnan(S4_4h[i]) or
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3 = R3_4h[i]
        s3 = S3_4h[i]
        r4 = R4_4h[i]
        s4 = S4_4h[i]
        ema_trend = ema_34_4h[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: bounce from S3 with uptrend and volume spike
            if price > s3 and price < s4 and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: rejection at R3 with downtrend and volume spike
            elif price < r3 and price > r4 and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price reaches R3 or breaks below EMA
            if price >= r3 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price reaches S3 or breaks above EMA
            if price <= s3 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_VolumeSpike_1dTrend_v6"
timeframe = "4h"
leverage = 1.0