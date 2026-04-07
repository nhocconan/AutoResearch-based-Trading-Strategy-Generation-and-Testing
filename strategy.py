#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily Pivot Breakout with Volume and ADX Trend Filter
# Hypothesis: Price breaking above/below daily pivot levels (R1/S1) with volume confirmation
# and ADX > 25 for trend strength works in both bull and bear markets.
# In bull markets: buy R1 breakouts, sell at R2.
# In bear markets: sell S1 breakdowns, cover at S2.
# Uses daily pivot points calculated from prior day's OHLC.
# Target: 20-40 trades/year (80-160 over 4 years).

name = "6h_daily_pivot_breakout_volume_adx_v1"
timeframe = "6h"
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
    if len(df_daily) < 1:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day's OHLC
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (daily_high + daily_low + daily_close) / 3.0
    range_hl = daily_high - daily_low
    
    # Standard pivot levels: R1, R2, S1, S2
    r1 = pivot + range_hl
    s1 = pivot - range_hl
    r2 = pivot + 2 * range_hl
    s2 = pivot - 2 * range_hl
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    pivot = np.roll(pivot, 1)
    r1 = np.roll(r1, 1)
    r2 = np.roll(r2, 1)
    s1 = np.roll(s1, 1)
    s2 = np.roll(s2, 1)
    
    # Handle first element
    if len(pivot) > 1:
        pivot[0] = pivot[1]
        r1[0] = r1[1]
        r2[0] = r2[1]
        s1[0] = s1[1]
        s2[0] = s2[1]
    else:
        pivot[0] = 0
        r1[0] = 0
        r2[0] = 0
        s1[0] = 0
        s2[0] = 0
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    
    # ADX filter for trend strength (14-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up = high_series.diff()
    down = low_series.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    plus_dm = pd.Series(plus_dm, index=high_series.index)
    minus_dm = pd.Series(minus_dm, index=low_series.index)
    
    # Smoothed DM
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(adx_values[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit at R2 or if trend weakens
            if high[i] >= r2_aligned[i] or adx_values[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit at S2 or if trend weakens
            if low[i] <= s2_aligned[i] or adx_values[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: break above R1 with volume and trend
            if high[i] > r1_aligned[i] and close[i] > r1_aligned[i] and \
               adx_values[i] > 25 and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: break below S1 with volume and trend
            elif low[i] < s1_aligned[i] and close[i] < s1_aligned[i] and \
                 adx_values[i] > 25 and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals