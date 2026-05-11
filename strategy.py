#!/usr/bin/env python3
"""
1d_1w_Camarilla_R3S3_Breakout_Trend
Hypothesis: Use weekly trend filter with Camarilla R3/S3 breakout on daily.
Long when price breaks above R3 and weekly trend is up; short when breaks below S3 and weekly trend is down.
Exit when price crosses back to H4 or L4 levels.
Camarilla levels provide institutional support/resistance. Weekly trend filter avoids counter-trend trades.
Targets 10-20 trades/year (40-80 over 4 years) to minimize fee drag.
"""

name = "1d_1w_Camarilla_R3S3_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily Camarilla levels: calculate from previous day's range
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # H4 = close + 1.1*(high-low), L4 = close - 1.1*(high-low)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Avoid using first bar
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else prev_high[0]
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else prev_low[0]
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else prev_close[0]
    
    range_hl = prev_high - prev_low
    r3 = prev_close + 1.1 * range_hl / 2
    s3 = prev_close - 1.1 * range_hl / 2
    h4 = prev_close + 1.1 * range_hl
    l4 = prev_close - 1.1 * range_hl
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 35  # for weekly EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(h4[i]) or np.isnan(l4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close[i] > ema34_1w_aligned[i]
        trend_down = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend
            if close[i] > r3[i] and trend_up:
                # Long: price breaks above R3 and weekly trend up
                signals[i] = 0.25
                position = 1
            elif close[i] < s3[i] and trend_down:
                # Short: price breaks below S3 and weekly trend down
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses back to H4/L4 or trend reverses
            if position == 1:
                # Exit long: price crosses below H4 OR trend turns down
                if close[i] < h4[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above L4 OR trend turns up
                if close[i] > l4[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals