#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly EMA Trend + Daily Range Breakout with Volume Confirmation
# Uses weekly EMA50 trend direction for bias, daily range breakout above/below
# previous day's high/low for entry, and volume confirmation (>1.5x average).
# Designed to capture trending moves while avoiding choppy markets.
# Target: 15-25 trades/year on 1d timeframe.

name = "1d_WeeklyEMA50_DailyRangeBreakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
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
    
    # Align weekly EMA50 to daily timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate daily average volume for volume confirmation
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range breakout: today's high > yesterday's high OR today's low < yesterday's low
        high_breakout = high[i] > high[i-1]
        low_breakout = low[i] < low[i-1]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: follow weekly EMA trend with range breakout and volume
            # Long when price above weekly EMA50 and breaks above previous day's high
            long_condition = (
                close[i] > ema50_weekly_aligned[i] and   # price above weekly EMA50 (bullish bias)
                high_breakout and                        # broke above previous day's high
                vol_confirm                              # volume confirmation
            )
            
            # Short when price below weekly EMA50 and breaks below previous day's low
            short_condition = (
                close[i] < ema50_weekly_aligned[i] and   # price below weekly EMA50 (bearish bias)
                low_breakout and                         # broke below previous day's low
                vol_confirm                              # volume confirmation
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below weekly EMA50 or loses momentum
            if close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above weekly EMA50 or loses momentum
            if close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals