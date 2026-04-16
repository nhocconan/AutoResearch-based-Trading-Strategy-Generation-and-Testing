#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate ATR on 6h
    tr_6h = np.maximum(high_6h - low_6h,
                       np.maximum(np.abs(high_6h - np.roll(close_6h, 1)),
                                  np.abs(low_6h - np.roll(close_6h, 1))))
    tr_6h[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Weekly Pivot Points (based on prior week) ===
    # For each day, we need the pivot from the previous week
    # We'll calculate weekly OHLC then derive pivots
    # Simplified: use daily close to approximate weekly pivot (acceptable for 6h)
    # Better: calculate actual weekly pivot from prior week's daily data
    
    # Calculate weekly high/low/close using 1d data grouped by week
    # Create a week number for each daily bar
    days = pd.to_datetime(df_1d.index if hasattr(df_1d, 'index') else range(len(df_1d)))
    if hasattr(df_1d, 'index'):
        week_numbers = days.isocalendar().week
    else:
        # Fallback: approximate week number
        week_numbers = (np.arange(len(df_1d)) // 7).astype(int)
    
    # Calculate weekly OHLC
    weekly_high = np.full(len(df_1d), np.nan)
    weekly_low = np.full(len(df_1d), np.nan)
    weekly_close = np.full(len(df_1d), np.nan)
    
    for week in np.unique(week_numbers[~np.isnan(week_numbers)]):
        mask = (week_numbers == week)
        if np.any(mask):
            weekly_high[mask] = np.max(high_1d[mask])
            weekly_low[mask] = np.min(low_1d[mask])
            weekly_close[mask] = close_1d[mask][-1]  # last day of week
    
    # Weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivots to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # === 1d ADX for trend strength ===
    # Calculate +DM, -DM, TR
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_1d_wilder = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_1d_wilder
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_1d_wilder
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Williams %R for overbought/oversold ===
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    # Handle division by zero
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        adx_val = adx_1d_aligned[i]
        wr = williams_r[i]
        weekly_r1 = weekly_r1_aligned[i]
        weekly_s1 = weekly_s1_aligned[i]
        weekly_r2 = weekly_r2_aligned[i]
        weekly_s2 = weekly_s2_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price crosses below weekly S1 OR Williams %R exits overbought
            if (price < weekly_s1) or (wr > -20):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above weekly R1 OR Williams %R exits oversold
            if (price > weekly_r1) or (wr < -80):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Strong trend filter (ADX > 25)
            if adx_val > 25:
                # LONG: Price above weekly R1 AND Williams %R not overbought
                if (price > weekly_r1) and (wr > -80):
                    signals[i] = 0.25
                    position = 1
                    continue
                # SHORT: Price below weekly S1 AND Williams %R not oversold
                elif (price < weekly_s1) and (wr < -20):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_ADX_WilliamsR"
timeframe = "6h"
leverage = 1.0