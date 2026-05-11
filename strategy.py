#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_Volume_Spike
Hypothesis: Uses daily Camarilla pivot levels with weekly trend filter (ADX>25) and volume spike confirmation. 
Works in bull markets via breakouts above R3 and in bear markets via breakdowns below S3, with volume confirmation ensuring institutional participation. 
Weekly ADX filter avoids whipsaws in sideways markets. Target: 15-25 trades/year to minimize fee drag while capturing strong monthly moves.
"""

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Camarilla calculation (based on previous day)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First day values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Pivot point and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + (range_val * 1.1 / 2)
    S3 = pivot - (range_val * 1.1 / 2)
    R4 = pivot + (range_val * 1.1)
    S4 = pivot - (range_val * 1.1)
    
    # Weekly ADX for trend filter
    # Calculate True Range
    tr1 = pd.Series(df_1w['high']).subtract(df_1w['low']).abs()
    tr2 = pd.Series(df_1w['high']).subtract(df_1w['close'].shift(1)).abs()
    tr3 = pd.Series(df_1w['low']).subtract(df_1w['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = pd.Series(df_1w['high']).diff()
    dm_minus = pd.Series(df_1w['low']).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_1w = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    
    # Calculate DX and ADX
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx_1w = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1w_values = adx_1w.values
    
    # Align ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w_values)
    
    # Volume spike detection (2x 20-day average)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3[i]) or 
            np.isnan(S3[i]) or
            np.isnan(adx_1w_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime based on weekly ADX
        adx = adx_1w_aligned[i]
        is_trending = adx > 25
        
        # Breakout signals
        long_breakout = (high[i] > R3[i]) and vol_spike[i]
        short_breakout = (low[i] < S3[i]) and vol_spike[i]
        
        if position == 0:
            # Only take trades in trending markets (ADX > 25)
            if is_trending:
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches S3 or weekly trend weakens
                exit_signal = (low[i] < S3[i]) or (adx < 20)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches R3 or weekly trend weakens
                exit_signal = (high[i] > R3[i]) or (adx < 20)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals