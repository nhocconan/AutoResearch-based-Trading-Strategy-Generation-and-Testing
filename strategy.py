# 1d_1w_ChannelBreakout_TrendFilter
# Hypothesis: Price breaking above/below the weekly Donchian channel with daily trend filter and volume
# confirmation captures momentum moves while reducing false signals. The weekly channel provides
# strong support/resistance levels, the daily trend filter ensures alignment with higher timeframe
# momentum, and volume confirmation adds conviction. Designed for low frequency via 1d timeframe
# and strict entry criteria to avoid overtrading.
# Target: 30-100 total trades over 4 years (7-25/year).

#!/usr/bin/env python3
name = "1d_1w_ChannelBreakout_TrendFilter"
timeframe = "1d"
leverage = 1.0

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
    
    # Weekly Donchian Channel (20-period)
    period = 20
    # Calculate weekly high and low using rolling window
    weekly_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    weekly_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period - 1)  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly high + weekly uptrend + volume
            if close[i] > weekly_high[i] and close[i] > ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low + weekly downtrend + volume
            elif close[i] < weekly_low[i] and close[i] < ema_50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to the weekly Donchian midpoint (mean reversion)
            weekly_mid = (weekly_high[i] + weekly_low[i]) / 2
            if position == 1:
                if close[i] <= weekly_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= weekly_mid:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals