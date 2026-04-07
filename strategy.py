#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h 12h/1d Pivot Breakout with Volume and Choppiness Filter
# Hypothesis: Daily and 12h pivot levels act as strong support/resistance. 
# Price breaking above R1 with volume indicates institutional buying, leading to continuation. 
# Price breaking below S1 with volume indicates institutional selling, leading to continuation. 
# Choppiness filter (CHOP > 61.8) ensures we only trade in ranging markets where pivot reversals work.
# Works in both bull and bear markets because: In bull, breaks above R1 continue up; breaks below S1 get bought (mean reversion). 
# In bear, breaks below S1 continue down; breaks above R1 get sold (mean reversion). 
# Volume filter ensures only institutional participation triggers entries.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_12h_1d_pivot_breakout_volume_chop_v3"
timeframe = "4h"
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
    
    # Get daily and 12h data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_daily) < 2 or len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate daily data (previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate 12h data (previous 12h bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Shift by 1 to use previous day's/12h bar's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high[0] = prev_daily_high[1] if len(prev_daily_high) > 1 else 0
    prev_daily_low[0] = prev_daily_low[1] if len(prev_daily_low) > 1 else 0
    prev_daily_close[0] = prev_daily_close[1] if len(prev_daily_close) > 1 else 0
    
    prev_12h_high = np.roll(high_12h, 1)
    prev_12h_low = np.roll(low_12h, 1)
    prev_12h_close = np.roll(close_12h, 1)
    prev_12h_high[0] = prev_12h_high[1] if len(prev_12h_high) > 1 else 0
    prev_12h_low[0] = prev_12h_low[1] if len(prev_12h_low) > 1 else 0
    prev_12h_close[0] = prev_12h_close[1] if len(prev_12h_close) > 1 else 0
    
    # Calculate daily pivot points
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_r1 = (2 * daily_pivot) - prev_daily_low
    daily_s1 = (2 * daily_pivot) - prev_daily_high
    daily_r2 = daily_pivot + (prev_daily_high - prev_daily_low)
    daily_s2 = daily_pivot - (prev_daily_high - prev_daily_low)
    
    # Calculate 12h pivot points
    pivot_12h = (prev_12h_high + prev_12h_low + prev_12h_close) / 3.0
    r1_12h = (2 * pivot_12h) - prev_12h_low
    s1_12h = (2 * pivot_12h) - prev_12h_high
    r2_12h = pivot_12h + (prev_12h_high - prev_12h_low)
    s2_12h = pivot_12h - (prev_12h_high - prev_12h_low)
    
    # Align to 4h timeframe (use previous day's/12h bar's levels)
    daily_pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    daily_r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    daily_r2_aligned = align_htf_to_ltf(prices, df_daily, daily_r2)
    daily_s2_aligned = align_htf_to_ltf(prices, df_daily, daily_s2)
    
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Choppiness filter: avoid trending markets (CHOP < 38.2)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI (14)
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1  # negative of downward move
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM and -DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    # Avoid division by zero
    chop = 100 * np.log10(atr_sum / (range_14 + 1e-10)) / np.log10(14)
    chop[range_14 < 1e-10] = 50  # Neutral when range is zero
    
    # Chop > 61.8 = ranging market (good for mean reversion at pivots)
    chop_filter = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(daily_pivot_aligned[i]) or np.isnan(daily_r1_aligned[i]) or 
            np.isnan(daily_s1_aligned[i]) or np.isnan(daily_r2_aligned[i]) or 
            np.isnan(daily_s2_aligned[i]) or np.isnan(pivot_12h_aligned[i]) or 
            np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(r2_12h_aligned[i]) or np.isnan(s2_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to daily pivot or 12h pivot or filters fail
            if (close[i] <= daily_pivot_aligned[i] or close[i] <= pivot_12h_aligned[i] or 
                not vol_filter[i] or not chop_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to daily pivot or 12h pivot or filters fail
            if (close[i] >= daily_pivot_aligned[i] or close[i] >= pivot_12h_aligned[i] or 
                not vol_filter[i] or not chop_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Only trade in ranging markets (CHOP > 61.8)
            if chop_filter[i]:
                # Long: price breaks above daily R1 or 12h R1 with volume
                if ((high[i] > daily_r1_aligned[i] or high[i] > r1_12h_aligned[i]) and 
                    (close[i] > daily_r1_aligned[i] or close[i] > r1_12h_aligned[i]) and 
                    vol_filter[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below daily S1 or 12h S1 with volume
                elif ((low[i] < daily_s1_aligned[i] or low[i] < s1_12h_aligned[i]) and 
                      (close[i] < daily_s1_aligned[i] or close[i] < s1_12h_aligned[i]) and 
                      vol_filter[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals