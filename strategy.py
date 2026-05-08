#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 breakout with 1d trend filter and volume spike
# Camarilla pivots provide strong support/resistance levels. Breakout above R3 or below S3
# with volume confirmation and aligned daily trend (EMA34) captures strong moves.
# Designed for low trade frequency: only trades on significant breakouts with confirmation.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    # Typical price for the period
    typical_price = (high + low + close) / 3.0
    # Range
    range_val = high - low
    
    # Camarilla levels
    # R4 = close + (high - low) * 1.500
    # R3 = close + (high - low) * 1.250
    # R2 = close + (high - low) * 1.166
    # R1 = close + (high - low) * 1.083
    # S1 = close - (high - low) * 1.083
    # S2 = close - (high - low) * 1.166
    # S3 = close - (high - low) * 1.250
    # S4 = close - (high - low) * 1.500
    
    r3 = close + range_val * 1.250
    s3 = close - range_val * 1.250
    r4 = close + range_val * 1.500
    s4 = close - range_val * 1.500
    
    return r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels for each 12h bar
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(n):
        r3[i], s3[i], r4[i], s4[i] = calculate_camarilla(high[i], low[i], close[i])
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        r3_val = r3[i]
        s3_val = s3[i]
        r4_val = r4[i]
        s4_val = s4[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike and uptrend
            if (close[i] > r3_val and vol_spike and close[i] > ema34_1d_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike and downtrend
            elif (close[i] < s3_val and vol_spike and close[i] < ema34_1d_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (reversal) or trend changes
            if close[i] < s3_val or close[i] < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (reversal) or trend changes
            if close[i] > r3_val or close[i] > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals