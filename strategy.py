#!/usr/bin/env python3
"""
1d_WeeklyTrend_DailyBreakout_12hVolume
Hypothesis: Weekly trend (using 20-week EMA on 1w) filters direction, daily breakout above/below 
previous day's high/low (breakout of daily range) with volume confirmation (12h volume > 1.5x 20-period 
average) provides edge in both bull and bear markets by capturing momentum after consolidation.
Timeframe: 1d for fewer trades, lower fee drag. Uses 1w trend filter to avoid counter-trend trades.
Target: 15-25 trades/year (~60-100 total over 4 years) to stay within fee limits.
"""

name = "1d_WeeklyTrend_DailyBreakout_12hVolume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1w data for trend filter (20-week EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly close
    ema20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema20_1w[i] = (close_1w[i] * 2 + ema20_1w[i-1] * 18) / 20
    
    # Align 1w EMA20 to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period average volume on 12h
    vol_ma_12h = np.full_like(volume_12h, np.nan)
    if len(volume_12h) >= 20:
        vol_ma_12h[19] = np.mean(volume_12h[0:20])
        for i in range(20, len(volume_12h)):
            vol_ma_12h[i] = (vol_ma_12h[i-1] * 19 + volume_12h[i]) / 20
    
    # Align 12h volume MA to daily timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Daily range breakout: today's high > yesterday's high (for long) or today's low < yesterday's low (for short)
    # We need yesterday's high/low, so shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # First bar has no previous, set to nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data: need EMA20_1w aligned, volume MA aligned, and prev high/low
    start_idx = max(20, 20)  # EMA20 and vol MA both need 20 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(prev_high[i]) or np.isnan(prev_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine conditions
        # Weekly trend: price above/below 20-week EMA
        trend_up = close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema20_1w_aligned[i]
        
        # Daily breakout: today breaks above yesterday's high or below yesterday's low
        breakout_up = high[i] > prev_high[i]
        breakout_down = low[i] < prev_low[i]
        
        # Volume confirmation: 12h volume > 1.5x its 20-period average
        # Note: vol_ma_12h_aligned gives the 20-period MA of 12h volume, aligned to daily
        volume_surge = volume[i] > (vol_ma_12h_aligned[i] * 1.5)
        
        if position == 0:
            # Enter long: uptrend + upward breakout + volume surge
            if trend_up and breakout_up and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend + downward breakout + volume surge
            elif trend_down and breakout_down and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend turns down OR price breaks below yesterday's low (mean reversion)
            if not trend_up or low[i] < prev_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend turns up OR price breaks above yesterday's high
            if trend_down or high[i] > prev_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals