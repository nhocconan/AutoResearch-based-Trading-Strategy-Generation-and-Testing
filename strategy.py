# I've been analyzing the pattern of failures and the successful strategies from the research.
# The key insights are:
# 1. Strategies need to generate very few trades (20-50 total over 4 years for 1d timeframe) to avoid fee drag
# 2. Successful strategies use price channels (Donchian, Keltner) or pivot levels combined with volume confirmation
# 3. Trend filters from higher timeframes help avoid counter-trend trades
# 4. Discrete position sizing (0.0, ±0.25) minimizes churn
# 
# This strategy combines:
# - 1-week Donchian channel breakout as the primary signal (very few signals naturally)
# - 1-day EMA(50) trend filter to ensure we trade with the higher timeframe trend
# - Volume confirmation (current volume > 1.5x 20-period average) to avoid false breakouts
# - Discrete position sizing of 0.25 to manage risk
# 
# The weekly Donchian breakout is inherently infrequent, which should keep trade counts low.
# The trend filter ensures we only take trades in the direction of the daily trend.
# Volume confirmation helps avoid false breakouts.

#!/usr/bin/env python3
name = "1d_WeeklyDonchian_1dTrend_Filter"
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
    
    # Get weekly data for Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period Donchian channels on weekly data
    # Upper band = highest high over past 20 weeks
    # Lower band = lowest low over past 20 weeks
    high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    # Align weekly Donchian bands to daily timeframe
    # We only want to act on signals after the weekly bar has closed
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Align daily trend filter
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper band + daily uptrend + volume confirmation
            if close[i] > high_20w_aligned[i] and trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian lower band + daily downtrend + volume confirmation
            elif close[i] < low_20w_aligned[i] and not trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below weekly Donchian lower band OR daily trend turns down
            if close[i] < low_20w_aligned[i] or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above weekly Donchian upper band OR daily trend turns up
            if close[i] > high_20w_aligned[i] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals