#!/usr/bin/env python3
# 1h_camarilla_pivot_volume_v1
# Hypothesis: 1h strategy using daily Camarilla pivot levels with volume confirmation for entry timing.
# Uses 1d Camarilla pivots for structure (R3/S3 levels) and 4h EMA for trend filter.
# Long: Price breaks above R3 with volume > 1.5x 20-period average AND price > 4h EMA20.
# Short: Price breaks below S3 with volume > 1.5x 20-period average AND price < 4h EMA20.
# Exit: Price returns to pivot point (PP).
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Position size: 0.20 (discrete to minimize fee churn).
# Target: 15-37 trades/year (60-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_camarilla_pivot_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r3 = pp + (range_1d * 3.0 / 8.0)
    
    # Support levels
    s3 = pp - (range_1d * 3.0 / 8.0)
    
    # Align Camarilla levels to 1h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # EMA20 on 4h close
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long entry: Price breaks above R3 with volume confirmation AND price > 4h EMA20 (uptrend)
            if (close[i] > r3_aligned[i] and volume_confirmed and close[i] > ema_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: Price breaks below S3 with volume confirmation AND price < 4h EMA20 (downtrend)
            elif (close[i] < s3_aligned[i] and volume_confirmed and close[i] < ema_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals