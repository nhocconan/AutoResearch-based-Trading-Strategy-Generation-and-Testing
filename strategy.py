#!/usr/bin/env python3
"""
1d_Weekly_Range_Reversal_v1
Hypothesis: Uses weekly pivot points (based on prior week's range) on daily timeframe to identify mean-reversion opportunities in both bull and bear markets. Combines with daily volume confirmation and ADX trend filter to avoid false signals. Target: 10-20 trades/year to minimize fee drag while capturing reversal opportunities at key weekly levels.
"""

name = "1d_Weekly_Range_Reversal_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Weekly Pivot Points (based on prior week) ---
    # Calculate pivot points from previous week's data
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Support and Resistance levels
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # --- Daily ADX for trend filter (avoid trading against strong trends) ---
    # Calculate True Range
    tr1 = pd.Series(high).subtract(pd.Series(low)).abs()
    tr2 = pd.Series(high).subtract(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).subtract(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = pd.Series(high).diff()
    dm_minus = pd.Series(low).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_daily = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_daily
    di_minus = 100 * dm_minus_smooth / atr_daily
    
    # Calculate DX and ADX
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx_daily = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_daily_values = adx_daily.values
    
    # --- Volume Spike Detection ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)  # Volume above 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(adx_daily_values[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime based on ADX
        adx = adx_daily_values[i]
        is_strong_trend = adx > 25  # Avoid mean reversion in strong trends
        
        # Mean reversion signals at weekly support/resistance levels
        # Long when price touches weekly support with volume spike
        long_signal = (
            (low[i] <= s1_aligned[i] or low[i] <= s2_aligned[i]) and 
            vol_spike[i] and 
            not is_strong_trend
        )
        
        # Short when price touches weekly resistance with volume spike
        short_signal = (
            (high[i] >= r1_aligned[i] or high[i] >= r2_aligned[i]) and 
            vol_spike[i] and 
            not is_strong_trend
        )
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to pivot or opposite S/R level
            if position == 1:
                # Exit long: price reaches pivot or resistance
                exit_signal = (
                    high[i] >= pivot_aligned[i] or 
                    high[i] >= r1_aligned[i]
                )
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches pivot or support
                exit_signal = (
                    low[i] <= pivot_aligned[i] or 
                    low[i] <= s1_aligned[i]
                )
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals