#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Pivot Breakout with Volume and ADX Filter
# Hypothesis: Daily pivot levels act as strong support/resistance. 
# Price breaking above R1 or below S1 with volume and trending conditions (ADX>25) 
# indicates institutional participation and continuation. 
# In bull: breaks above R1 continue up; breaks below S1 get bought (mean reversion)
# In bear: breaks below S1 continue down; breaks above R1 get sold (mean reversion)
# Uses volume and ADX filters to confirm institutional participation and trend strength.
# Target: 12-30 trades/year (48-120 over 4 years).

name = "12h_daily_pivot_breakout_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily data (previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high[0] = prev_daily_high[1] if len(prev_daily_high) > 1 else 0
    prev_daily_low[0] = prev_daily_low[1] if len(prev_daily_low) > 1 else 0
    prev_daily_close[0] = prev_daily_close[1] if len(prev_daily_close) > 1 else 0
    
    # Calculate daily pivot points
    # Pivot = (High + Low + Close) / 3
    # R1 = (2 * Pivot) - Low
    # S1 = (2 * Pivot) - High
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_r1 = (2 * daily_pivot) - prev_daily_low
    daily_s1 = (2 * daily_pivot) - prev_daily_high
    
    # Align to 12h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ADX filter: ADX > 25 indicates trending market
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR and DM
    tr_period = 14
    tr_sum = np.zeros(n)
    dm_plus_sum = np.zeros(n)
    dm_minus_sum = np.zeros(n)
    
    # Initial values
    tr_sum[tr_period] = np.nansum(tr[1:tr_period+1])
    dm_plus_sum[tr_period] = np.nansum(dm_plus[1:tr_period+1])
    dm_minus_sum[tr_period] = np.nansum(dm_minus[1:tr_period+1])
    
    # Wilder smoothing
    for i in range(tr_period + 1, n):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # Calculate DI and DX
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)
    
    for i in range(tr_period, n):
        if tr_sum[i] != 0:
            di_plus[i] = 100 * (dm_plus_sum[i] / tr_sum[i])
            di_minus[i] = 100 * (dm_minus_sum[i] / tr_sum[i])
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # Smoothed ADX
    adx = np.zeros(n)
    adx_period = 14
    if n >= 2 * adx_period:
        adx[2*adx_period] = np.nanmean(dx[adx_period:2*adx_period+1])
        for i in range(2*adx_period + 1, n):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to pivot or volume drops or ADX weakens
            if close[i] <= pivot_aligned[i] or not vol_filter[i] or not adx_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to pivot or volume drops or ADX weakens
            if close[i] >= pivot_aligned[i] or not vol_filter[i] or not adx_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R1 with volume and trend
            if high[i] > r1_aligned[i] and close[i] > r1_aligned[i] and vol_filter[i] and adx_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S1 with volume and trend
            elif low[i] < s1_aligned[i] and close[i] < s1_aligned[i] and vol_filter[i] and adx_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals