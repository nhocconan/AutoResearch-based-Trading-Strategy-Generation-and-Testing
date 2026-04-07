#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Pivot Breakout with Volume Confirmation
# Hypothesis: Price breaking out of weekly pivot-based resistance/support levels (R4/S4)
# with volume confirmation and weekly trend filter (price vs weekly 20 EMA) captures
# strong momentum moves in both bull and bear markets. Weekly pivots act as
# institutional reference points, breakouts signal regime changes.
# In bull markets: buy breakouts above weekly R4 with volume.
# In bear markets: sell breakouts below weekly S4 with volume.
# Weekly timeframe reduces noise, volume confirms institutional participation.
# Target: 12-30 trades/year (50-120 over 4 years).

name = "6h_weekly_pivot_breakout_volume_v1"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly high, low, close for pivot points
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    weekly_p = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_p - weekly_low
    weekly_s1 = 2 * weekly_p - weekly_high
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_r2 = weekly_p + (weekly_high - weekly_low)
    weekly_s2 = weekly_p - (weekly_high - weekly_low)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_r3 = weekly_high + 2 * (weekly_p - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_p)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)  [Extended levels]
    weekly_r4 = weekly_r3 + (weekly_high - weekly_low)
    weekly_s4 = weekly_s3 - (weekly_high - weekly_low)
    
    # Shift by 1 to use only completed weekly bars (avoid look-ahead)
    weekly_r4 = np.roll(weekly_r4, 1)
    weekly_s4 = np.roll(weekly_s4, 1)
    weekly_p = np.roll(weekly_p, 1)  # for trend filter
    
    # Handle first element
    if len(weekly_r4) > 1:
        weekly_r4[0] = weekly_r4[1]
        weekly_s4[0] = weekly_s4[1]
        weekly_p[0] = weekly_p[1]
    else:
        weekly_r4[0] = 0
        weekly_s4[0] = 0
        weekly_p[0] = 0
    
    # Weekly trend filter: price vs weekly pivot (P)
    # In bull markets: price > P, favor longs
    # In bear markets: price < P, favor shorts
    # We'll use this as a bias filter
    
    # Align weekly data to 6h timeframe
    weekly_r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    weekly_p_aligned = align_htf_to_ltf(prices, df_weekly, weekly_p)
    
    # Volume filter: volume > 1.5x 20-period average (institutional participation)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or
            np.isnan(weekly_p_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly S4 or trend turns bearish (price < weekly pivot)
            if close[i] < weekly_s4_aligned[i] or close[i] < weekly_p_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above weekly R4 or trend turns bullish (price > weekly pivot)
            if close[i] > weekly_r4_aligned[i] or close[i] > weekly_p_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above weekly R4 with volume and bullish trend (price > pivot)
            if (high[i] > weekly_r4_aligned[i] and close[i] > weekly_r4_aligned[i] and
                close[i] > weekly_p_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below weekly S4 with volume and bearish trend (price < pivot)
            elif (low[i] < weekly_s4_aligned[i] and close[i] < weekly_s4_aligned[i] and
                  close[i] < weekly_p_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals