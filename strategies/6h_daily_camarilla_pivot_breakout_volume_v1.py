#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6H Daily Camarilla Pivot Breakout with Volume Confirmation
# Hypothesis: Camarilla pivot levels derived from daily OHLC provide reliable support/resistance.
# Breakouts beyond R4/S4 with volume confirmation capture strong momentum moves.
# Works in bull markets (buy R4 breakouts) and bear markets (sell S4 breakdowns).
# Daily timeframe provides robust levels, volume confirms institutional interest.
# Target: 12-37 trades/year (50-150 total over 4 years).

name = "6h_daily_camarilla_pivot_breakout_volume_v1"
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
    
    # Get daily data for Camarilla pivots
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily high/low/close
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    camarilla_r4 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_r3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_s3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    camarilla_s4 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Align daily Camarilla levels to 6h timeframe (shifted by 1 for completed bars)
    r4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s4)
    
    # Volume filter: volume > 1.8x 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below R3 (take profit) or reverses below S4 (stop)
            if close[i] < r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above S3 (take profit) or reverses above R4 (stop)
            if close[i] > s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above R4 with volume
            if (high[i] > r4_aligned[i] and close[i] > r4_aligned[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below S4 with volume
            elif (low[i] < s4_aligned[i] and close[i] < s4_aligned[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals