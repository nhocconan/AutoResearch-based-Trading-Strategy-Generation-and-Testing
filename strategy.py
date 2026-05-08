#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian Breakout with Weekly Trend Filter and Volume Spike
# Uses 20-period Donchian channels on 6h timeframe for breakout signals.
# Weekly EMA50 determines trend direction: only long when above, short when below.
# Volume spike (>2x 20-period average) confirms institutional participation.
# Designed to work in both bull and bear markets by following the weekly trend.
# Target: 60-120 total trades over 4 years (15-30/year) to stay within optimal range.

name = "6h_Donchian_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = (close_weekly[i] * 2 + ema50_weekly[i-1] * 48) / 50
    
    # Calculate 20-period average volume for volume filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Calculate Donchian channels (20-period high/low)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Align weekly EMA50 to 6h timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_avg_20[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        vol_filter = volume[i] > 2.0 * vol_avg_20[i]
        
        # Determine trend direction
        bullish_trend = close[i] > ema50_weekly_aligned[i]
        bearish_trend = close[i] < ema50_weekly_aligned[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with volume and trend filter
            # Long when price breaks above Donchian high in bullish trend
            long_condition = (
                close[i] > donchian_high[i] and  # breakout above upper band
                bullish_trend and                # only long in uptrend
                vol_filter                       # volume confirmation
            )
            
            # Short when price breaks below Donchian low in bearish trend
            short_condition = (
                close[i] < donchian_low[i] and   # breakout below lower band
                bearish_trend and                # only short in downtrend
                vol_filter                       # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian low or trend changes
            if close[i] < donchian_low[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian high or trend changes
            if close[i] > donchian_high[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals