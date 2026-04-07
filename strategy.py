#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Pivot Breakout with Volume and Confluence Filter
# Hypothesis: Weekly pivot levels provide strong support/resistance. Price breaking above R1 or below S1 with
# volume confirmation and alignment with monthly trend (via monthly close > SMA10) captures institutional moves.
# Works in bull via continuation, in bear via mean reversion from extreme levels.
# Target: 15-25 trades/year (60-100 over 4 years) to stay within optimal range.

name = "1d_weekly_pivot_breakout_volume_confluence_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly data (previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    prev_weekly_close[0] = prev_weekly_close[1] if len(prev_weekly_close) > 1 else 0
    
    # Calculate weekly pivot points
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_r1 = (2 * weekly_pivot) - prev_weekly_low
    weekly_s1 = (2 * weekly_pivot) - prev_weekly_high
    weekly_r2 = weekly_pivot + (prev_weekly_high - prev_weekly_low)
    weekly_s2 = weekly_pivot - (prev_weekly_high - prev_weekly_low)
    
    # Align to 1d timeframe (use previous week's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # Volume filter: volume > 1.8x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    # Monthly trend filter: monthly close > 10-period SMA
    df_monthly = get_htf_data(prices, '1M')
    if len(df_monthly) >= 2:
        monthly_close = df_monthly['close'].values
        monthly_close_shifted = np.roll(monthly_close, 1)
        monthly_close_shifted[0] = monthly_close_shifted[1] if len(monthly_close_shifted) > 1 else 0
        monthly_sma = pd.Series(monthly_close_shifted).rolling(window=10, min_periods=10).mean().values
        monthly_aligned = align_htf_to_ltf(prices, df_monthly, monthly_sma)
        monthly_filter = monthly_close_shifted > monthly_aligned  # monthly close above its SMA
    else:
        monthly_filter = np.ones(n, dtype=bool)  # default true if insufficient data
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i]) or
            (len(df_monthly) >= 2 and np.isnan(monthly_aligned[i]))):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to pivot or volume drops or monthly trend turns bearish
            if close[i] <= pivot_aligned[i] or not vol_filter[i] or not monthly_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to pivot or volume drops or monthly trend turns bullish
            if close[i] >= pivot_aligned[i] or not vol_filter[i] or monthly_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R1 with volume and monthly uptrend
            if high[i] > r1_aligned[i] and close[i] > r1_aligned[i] and vol_filter[i] and monthly_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S1 with volume and monthly downtrend
            elif low[i] < s1_aligned[i] and close[i] < s1_aligned[i] and vol_filter[i] and not monthly_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals