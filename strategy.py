#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    weekly = get_htf_data(prices, '1w')
    weekly_close = weekly['close'].values
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    
    # Calculate weekly EMA for trend direction (21 period)
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMA to daily timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, weekly, weekly_ema)
    
    # Get daily data for pivot levels
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily pivot levels (classic floor trader pivots)
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Align pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, daily, r1)
    s1_aligned = align_htf_to_ltf(prices, daily, s1)
    
    # Volume filter: current daily volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Trend filter: price above/below weekly EMA
    price_above_weekly_ema = close > weekly_ema_aligned
    price_below_weekly_ema = close < weekly_ema_aligned
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(weekly_ema_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter[i]:
            # Long conditions: price breaks above R1 with volume and above weekly EMA
            if close[i] > r1_aligned[i] and price_above_weekly_ema[i]:
                signals[i] = 0.25
            # Long conditions: price bounces from S1 with volume and above weekly EMA
            elif close[i] > s1_aligned[i] and close[i] < (pivot_aligned[i] * 1.02) and price_above_weekly_ema[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below S1 with volume and below weekly EMA
            elif close[i] < s1_aligned[i] and price_below_weekly_ema[i]:
                signals[i] = -0.25
            # Short conditions: price rejected at R1 with volume and below weekly EMA
            elif close[i] < r1_aligned[i] and close[i] > (pivot_aligned[i] * 0.98) and price_below_weekly_ema[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_Pivot_R1_S1_Breakout_WeeklyTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0