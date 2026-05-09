#!/usr/bin/env python3
# 1d_WeeklyTrend_DailyBreakout_With_Volume
# Hypothesis: Daily breakout above/below prior day's high/low with weekly trend filter and volume confirmation.
# Long when weekly trend up (weekly close > weekly SMA20) and price breaks above prior day's high with volume > 1.5x average.
# Short when weekly trend down (weekly close < weekly SMA20) and price breaks below prior day's low with volume > 1.5x average.
# Uses weekly trend to filter daily breakouts, reducing false signals in choppy markets. Target: 10-25 trades/year (40-100 total over 4 years).

name = "1d_WeeklyTrend_DailyBreakout_With_Volume"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA20 for trend filter
    sma20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        sma20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            sma20_1w[i] = (sma20_1w[i-1] * 19 + close_1w[i]) / 20
    
    # Align weekly SMA20 to daily timeframe
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # Prior day's high and low for breakout levels
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_high[0] = np.nan  # No prior day for first bar
    prior_low[0] = np.nan
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need weekly SMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma20_1w_aligned[i]) or np.isnan(prior_high[i]) or 
            np.isnan(prior_low[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close_1w[-1] > sma20_1w_aligned[i] if len(close_1w) > 0 else False  # Simplified: use current weekly close vs SMA
        # More robust: use the weekly close value aligned to current day
        # Since we don't have weekly close aligned, we'll use the trend from the weekly data
        # Alternative: determine trend based on whether current price is above/below weekly SMA
        trend_up = close[i] > sma20_1w_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + price breaks above prior day's high + volume confirmation
            if trend_up and close[i] > prior_high[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + price breaks below prior day's low + volume confirmation
            elif not trend_up and close[i] < prior_low[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price breaks below prior day's low
            if not trend_up or close[i] < prior_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price breaks above prior day's high
            if trend_up or close[i] > prior_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals