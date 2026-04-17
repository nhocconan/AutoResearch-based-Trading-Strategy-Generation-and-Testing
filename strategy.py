#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_CamTrend_Filter
Hypothesis: Camarilla pivot levels (R1, S1) act as strong support/resistance on 12h timeframe.
Breakouts above R1 or below S1 with volume confirmation and 1-week trend filter capture
institutional moves while avoiding false breakouts in chop. Works in bull (trend continuation)
and bear (mean reversion at extremes) by filtering with 1w trend.
Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for 12h (based on prior 12h bar)
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    # We need the prior 12h bar's OHLC
    
    # Get 12h data for pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels using prior 12h bar (no look-ahead)
    # R1 = prior_close + 1.1 * (prior_high - prior_low) / 12
    # S1 = prior_close - 1.1 * (prior_high - prior_low) / 12
    prior_high = np.roll(high_12h, 1)
    prior_low = np.roll(low_12h, 1)
    prior_close = np.roll(close_12h, 1)
    # Set first value to NaN (no prior bar)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    rang = prior_high - prior_low
    R1 = prior_close + 1.1 * rang / 12
    S1 = prior_close - 1.1 * rang / 12
    
    # Align 12h levels to 12h timeframe (already in 12h bars)
    # We need to align to our trading timeframe (12h)
    # Since we're trading on 12h timeframe, we can use the values directly
    # but we need to expand to match the length of prices array
    
    # For 12h timeframe, prices array IS the 12h data (each row is a 12h bar)
    # So we can use R1 and S1 directly, but need to handle the roll
    
    # Actually, since we're using 12h timeframe, prices IS the 12h data
    # So we can simplify: use prior 12h bar's data from prices itself
    high_12h = high  # already 12h data
    low_12h = low
    close_12h = close
    
    prior_high = np.roll(high_12h, 1)
    prior_low = np.roll(low_12h, 1)
    prior_close = np.roll(close_12h, 1)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    rang = prior_high - prior_low
    R1 = prior_close + 1.1 * rang / 12
    S1 = prior_close - 1.1 * rang / 12
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1-week data for trend filter (major trend direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        # Fallback to 1d if 1w not available
        df_1w = get_htf_data(prices, '1d')
    close_1w = df_1w['close'].values
    
    # Calculate 1-week EMA50 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema50_1w = close_series_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1-week EMA to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 1)  # volume MA20, need at least 1 for prior bar
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(volume_ma20[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Breakout conditions
        breakout_long = close[i] > R1[i]  # Price breaks above R1
        breakout_short = close[i] < S1[i]  # Price breaks below S1
        
        if position == 0:
            # Long: breakout above R1 + volume filter + 1w uptrend (price > EMA50)
            if breakout_long and volume_filter and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + 1w downtrend (price < EMA50)
            elif breakout_short and volume_filter and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 (mean reversion) or opposite breakout
            if close[i] < S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 (mean reversion) or opposite breakout
            if close[i] > R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_CamTrend_Filter"
timeframe = "12h"
leverage = 1.0