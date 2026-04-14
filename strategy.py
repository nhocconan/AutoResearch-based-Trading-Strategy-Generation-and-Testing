#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Daily Pivot Point with Volume Confirmation and Trend Filter
# Uses daily pivot levels (R1/S1, R2/S2, R3/S3) from previous day as support/resistance
# Long when price bounces from S1/S2 with volume confirmation and uptrend filter
# Short when price is rejected from R1/R2 with volume confirmation and downtrend filter
# Volume filter requires current volume > 1.5x 20-period average to confirm institutional interest
# Trend filter uses 12h EMA(50) to align with higher timeframe momentum
# Designed to work in both bull and bear markets by fading extremes with institutional validation
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align pivot levels to 6h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 12h EMA (50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume_filter[i]
        
        # Trend filter: only trade in direction of 12h EMA
        above_ema = price > ema_12h_aligned[i]
        
        if position == 0:
            # Long: price bounces from S1/S2 with volume and uptrend
            if vol_ok and above_ema:
                if price <= s1_aligned[i] * 1.002 and price >= s1_aligned[i] * 0.998:
                    position = 1
                    signals[i] = position_size
                elif price <= s2_aligned[i] * 1.002 and price >= s2_aligned[i] * 0.998:
                    position = 1
                    signals[i] = position_size
            # Short: price rejected from R1/R2 with volume and downtrend
            elif vol_ok and not above_ema:
                if price >= r1_aligned[i] * 0.998 and price <= r1_aligned[i] * 1.002:
                    position = -1
                    signals[i] = -position_size
                elif price >= r2_aligned[i] * 0.998 and price <= r2_aligned[i] * 1.002:
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches pivot or trend changes
            if price >= pivot_aligned[i] or price < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches pivot or trend changes
            if price <= pivot_aligned[i] or price > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_DailyPivot_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0