#!/usr/bin/env python3
"""
12h_WeeklyTrend_Filtered_Camarilla_Breakout
Strategy: 12h Camarilla pivot breakout with weekly trend filter and volume confirmation.
Long: Price breaks above R1 (daily) + weekly close > weekly open + volume > 1.5x 20-period MA
Short: Price breaks below S1 (daily) + weekly close < weekly open + volume > 1.5x 20-period MA
Exit: Price returns to PP (pivot point) or volume drops below average
Position size: 0.25
Designed to capture institutional breakouts aligned with weekly trend, avoiding false breakouts in chop.
Timeframe: 12h
"""

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
    
    # Calculate daily Camarilla levels (using prior day's OHLC)
    # We'll compute daily OHLC from the price data by resampling conceptually
    # But per rules, we must use get_htf_data for actual daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day: based on prior day's OHLC
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # PP = (H+L+C)/3
    # We shift by 1 to use prior day's levels for current day's breakout
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's levels (shifted by 1)
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    # First day has no prior, set to zeros so levels are invalid
    prior_high[0] = 0
    prior_low[0] = 0
    prior_close[0] = 0
    
    # Calculate Camarilla levels based on prior day
    R1 = prior_close + (prior_high - prior_low) * 1.1 / 12
    S1 = prior_close - (prior_high - prior_low) * 1.1 / 12
    PP = (prior_high + prior_low + prior_close) / 3
    
    # Align to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    PP_12h = align_htf_to_ltf(prices, df_1d, PP)
    
    # Weekly trend filter: weekly close > weekly open = uptrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        # Fallback to daily if weekly not available
        weekly_uptrend = np.ones(len(prices))  # neutral
        weekly_downtrend = np.zeros(len(prices))
    else:
        weekly_open = df_1w['open'].values
        weekly_close = df_1w['close'].values
        weekly_uptrend = (weekly_close > weekly_open).astype(float)
        weekly_downtrend = (weekly_close < weekly_open).astype(float)
        weekly_uptrend = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
        weekly_downtrend = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Volume confirmation: 20-period MA on 12h
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # need volume MA and at least one prior day
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or np.isnan(PP_12h[i]) or
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Entry conditions
        if position == 0:
            # Long: break above R1 + weekly uptrend + volume
            if (close[i] > R1_12h[i] and weekly_uptrend[i] > 0.5 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + weekly downtrend + volume
            elif (close[i] < S1_12h[i] and weekly_downtrend[i] > 0.5 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to PP or volume fails
            if close[i] < PP_12h[i] or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to PP or volume fails
            if close[i] > PP_12h[i] or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyTrend_Filtered_Camarilla_Breakout"
timeframe = "12h"
leverage = 1.0