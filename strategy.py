#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume_confirmation
Hypothesis: 4-hour chart with daily Camarilla levels, volume confirmation, and volatility filter.
Uses tighter entry conditions (break of R4/S4 with volume > 1.5x 20-period average) to reduce trade frequency.
Designed for both bull and bear markets by using volatility-adjusted breakouts and volume confirmation to avoid false signals.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

name = "4h_1d_camarilla_breakout_volume_confirmation"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range (shifted by 1 to use completed day)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate range
    range_ = prev_high - prev_low
    
    # Camarilla levels (based on previous day)
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Volatility filter: avoid low volatility periods
    returns = np.diff(np.log(close), prepend=np.log(close)[0])
    volatility = pd.Series(returns).rolling(window=20, min_periods=20).std().values
    vol_filter = volatility > np.percentile(volatility[~np.isnan(volatility)], 20) if np.sum(~np.isnan(volatility)) > 0 else np.ones_like(volatility, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            not vol_confirm[i] or not vol_filter[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: close breaks above R4 with volume and volatility filter
        if close[i] > r4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below S4 with volume and volatility filter
        elif close[i] < s4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals