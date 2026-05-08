#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly EMA50 + Daily Price Action + Volume Confirmation
# Long when price crosses above Weekly EMA50 with volume > 1.5x 20-day average and closes in upper 50% of daily range
# Short when price crosses below Weekly EMA50 with volume > 1.5x 20-day average and closes in lower 50% of daily range
# Uses weekly trend filter to avoid counter-trend trades, volume surge for confirmation, and price action for entry quality
# Designed to work in both bull (buy dips) and bear (sell rallies) markets by following the weekly trend
# Target: 15-25 trades/year to stay within frequency limits and minimize fee drag

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
    
    # Get weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50)
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = (close_weekly[i] * 2 / (50 + 1)) + (ema50_weekly[i-1] * (49 / (50 + 1)))
    
    # Align weekly EMA50 to daily timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate 20-day average volume for volume filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate daily range position: (close - low) / (high - low)
        if high[i] == low[i]:
            range_pos = 0.5  # avoid division by zero
        else:
            range_pos = (close[i] - low[i]) / (high[i] - low[i])
        
        # Check volume condition: current volume > 1.5x 20-day average
        vol_condition = volume[i] > 1.5 * vol_avg_20[i]
        
        # Check trend condition: price relative to weekly EMA50
        price_above_ema = close[i] > ema50_weekly_aligned[i]
        price_below_ema = close[i] < ema50_weekly_aligned[i]
        
        if position == 0:
            # Look for entry: weekly trend + volume surge + price action
            if price_above_ema and vol_condition and range_pos > 0.5:
                signals[i] = 0.25
                position = 1
            elif price_below_ema and vol_condition and range_pos < 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below weekly EMA50 or loses momentum
            if close[i] < ema50_weekly_aligned[i] or range_pos < 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above weekly EMA50 or loses momentum
            if close[i] > ema50_weekly_aligned[i] or range_pos > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals