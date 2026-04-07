#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Pivot Breakout with Volume and Trend Filter
# Hypothesis: Price breaking above/below daily pivot levels with volume confirmation
# and trend filter (price vs 200 EMA) works in both bull and bear markets.
# In bull markets: long on breakout above R1, short on breakdown below S1.
# In bear markets: short on breakdown below S1, long on breakout above R1.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_pivot_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
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
    
    # Calculate daily pivot levels from previous day's OHLC
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (daily_high + daily_low + daily_close) / 3.0
    range_hl = daily_high - daily_low
    
    # Standard pivot levels (R1, S1, R2, S2)
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    r2 = pivot + range_hl
    s2 = pivot - range_hl
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    pivot = np.roll(pivot, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    r2 = np.roll(r2, 1)
    s2 = np.roll(s2, 1)
    
    # Handle first element
    if len(pivot) > 1:
        pivot[0] = pivot[1]
        r1[0] = r1[1]
        s1[0] = s1[1]
        r2[0] = r2[1]
        s2[0] = s2[1]
    else:
        pivot[0] = 0
        r1[0] = 0
        s1[0] = 0
        r2[0] = 0
        s2[0] = 0
    
    # Align to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    
    # Trend filter: price vs 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_200[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: breakdown below S1 or trend reversal
            if low[i] <= s1_aligned[i] or close[i] < ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit conditions: breakout above R1 or trend reversal
            if high[i] >= r1_aligned[i] or close[i] > ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above R1 with volume and trend
            if high[i] > r1_aligned[i] and close[i] > ema_200[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below S1 with volume and trend
            elif low[i] < s1_aligned[i] and close[i] < ema_200[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals