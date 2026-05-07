#!/usr/bin/env python3
"""
12h_DonchianBreakout_1dTrend_Volume
Hypothesis: Donchian channel breakouts on 12h timeframe, filtered by 1d EMA trend and volume spike, capture momentum moves with low trade frequency. Works in bull markets via breakouts and in bear markets via short breakdowns. Volume confirmation reduces false signals. 1d trend filter ensures alignment with higher timeframe momentum. Target: 50-150 total trades over 4 years.
"""
name = "12h_DonchianBreakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Donchian Channel (20-period)
    period = 20
    upper_channel = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower_channel = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 2.0 * 30-period average
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, period - 1)  # Need enough data for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper channel + 1d uptrend + volume spike
            if close[i] > upper_channel[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower channel + 1d downtrend + volume spike
            elif close[i] < lower_channel[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Donchian level (mean reversion)
            if position == 1:
                if close[i] < lower_channel[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if close[i] > upper_channel[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals