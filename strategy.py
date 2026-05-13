#!/usr/bin/env python3
"""
12h_1d_Camarilla_R3S3_Breakout_Trend_Volume
Hypothesis: Buy when price breaks above Camarilla R3 on 12h timeframe with 1d EMA34 trend filter and volume spike; sell when price breaks below S3 with 1d EMA34 downtrend and volume spike. Camarilla levels from daily pivot provide institutional support/resistance. EMA34 trend filter ensures trading with higher timeframe momentum. Volume spike confirms institutional interest. Designed for 12h timeframe to limit trades (target 50-150 total over 4 years) and avoid fee drag. Works in bull markets (breakouts continue) and bear markets (breakdowns continue) when aligned with trend.
"""

name = "12h_1d_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point
    pivot = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3.0
    # Calculate Camarilla levels for today (using yesterday's data)
    # We need to shift by 1 to avoid look-ahead: use previous day's OHLC for today's levels
    high_1d_shift = np.roll(high_1d, 1)
    low_1d_shift = np.roll(low_1d, 1)
    close_1d_shift = np.roll(close_1d, 1)
    # First element will be invalid, but we'll handle with min_periods in alignment
    pivot_shift = (high_1d_shift + low_1d_shift + close_1d_shift) / 3.0
    range_shift = high_1d_shift - low_1d_shift
    
    # Camarilla levels: R3 = pivot + 1.1 * range / 2, S3 = pivot - 1.1 * range / 2
    r3 = pivot_shift + 1.1 * range_shift / 2.0
    s3 = pivot_shift - 1.1 * range_shift / 2.0
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume average (24-period for ~12 days on 12h chart) for volume spike
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2.0x 24-period average
        vol_spike = volume[i] > 2.0 * vol_ma_24[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike and price above EMA34 (uptrend)
            if close[i] > r3_aligned[i] and vol_spike and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and price below EMA34 (downtrend)
            elif close[i] < s3_aligned[i] and vol_spike and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or EMA34 turns down
            if close[i] < s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or EMA34 turns up
            if close[i] > r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals