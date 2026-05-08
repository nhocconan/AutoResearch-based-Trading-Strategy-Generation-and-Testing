#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d combined with 1d EMA trend and volume confirmation
# - Fade at R3/S3 (mean reversion): short at R3, long at S3 when price touches these levels
# - Breakout continuation at R4/S4: long above R4, short below S4 when price breaks with momentum
# - Filtered by 1d EMA(34) trend direction to avoid counter-trend trades
# - Requires volume spike for confirmation to filter false signals
# - Designed for low frequency in both bull and bear markets by using daily pivots
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_Camarilla_R3S4_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val <= 0:
        return np.nan, np.nan, np.nan, np.nan
    c = close
    h = high
    l = low
    R4 = c + range_val * 1.1 / 2
    R3 = c + range_val * 1.1 / 4
    S3 = c - range_val * 1.1 / 4
    S4 = c - range_val * 1.1 / 2
    return R4, R3, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R4_1d, R3_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R4_1d_aligned[i]) or 
            np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or
            np.isnan(S4_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        r4_val = R4_1d_aligned[i]
        r3_val = R3_1d_aligned[i]
        s3_val = S3_1d_aligned[i]
        s4_val = S4_1d_aligned[i]
        close_price = close[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Fade at R3/S3: short at R3, long at S3 (mean reversion)
            if (close_price >= r3_val and close_price <= r4_val and 
                ema34_1d_val > close_price and  # slight downtrend bias for fade
                vol_spike):
                signals[i] = -0.25
                position = -1
            elif (close_price <= s3_val and close_price >= s4_val and 
                  ema34_1d_val < close_price and  # slight uptrend bias for fade
                  vol_spike):
                signals[i] = 0.25
                position = 1
            # Breakout continuation at R4/S4: break with momentum
            elif (close_price > r4_val and 
                  ema34_1d_val < close_price and  # uptrend
                  vol_spike):
                signals[i] = 0.25
                position = 1
            elif (close_price < s4_val and 
                  ema34_1d_val > close_price and  # downtrend
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 or breaks below S4 with volume
            if (close_price <= s3_val and vol_spike) or (close_price < s4_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 or breaks above R4 with volume
            if (close_price >= r3_val and vol_spike) or (close_price > r4_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals