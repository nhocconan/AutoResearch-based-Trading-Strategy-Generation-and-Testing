#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla Pivot + 1d Volume Spike + Trend Filter
# Uses daily Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout)
# Volume spike confirms institutional interest
# 1d EMA (50) filters trend direction to avoid counter-trend trades
# Camarilla levels provide institutional support/resistance
# Volume spike filters for meaningful moves
# Trend filter ensures alignment with higher timeframe momentum
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 2)
    s3 = pivot - (range_1d * 1.1 / 2)
    r4 = pivot + (range_1d * 1.1)
    s4 = pivot - (range_1d * 1.1)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d EMA (50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike detector (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Fade at R3/S3 with volume spike and trend alignment
            if (price >= r3_aligned[i] and price <= r4_aligned[i] and 
                vol_spike[i] and not above_ema):
                # Fade from R3 (sell high in uptrend context)
                position = -1
                signals[i] = -position_size
            elif (price <= s3_aligned[i] and price >= s4_aligned[i] and 
                  vol_spike[i] and above_ema):
                # Fade from S3 (buy low in downtrend context)
                position = 1
                signals[i] = position_size
            # Breakout at R4/S4 with volume spike and trend alignment
            elif price > r4_aligned[i] and vol_spike[i] and above_ema:
                # Breakout above R4 in uptrend
                position = 1
                signals[i] = position_size
            elif price < s4_aligned[i] and vol_spike[i] and not above_ema:
                # Breakdown below S4 in downtrend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches S3 (fade target) or trend changes
            if price <= s3_aligned[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches R3 (fade target) or trend changes
            if price >= r3_aligned[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Camarilla_VolumeSpike_TrendFilter"
timeframe = "6h"
leverage = 1.0