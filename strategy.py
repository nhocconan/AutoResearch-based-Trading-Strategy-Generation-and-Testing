#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channels (20-period)
    # Upper band: highest high of last 20 weeks
    high_series = pd.Series(high_1w)
    donchian_upper_20w = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 weeks
    low_series = pd.Series(low_1w)
    donchian_lower_20w = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(34) for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly volume average (20-period) for volume confirmation
    volume_series = pd.Series(volume_1w)
    volume_ma_20w = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (1.5 * volume_ma_20w)
    
    # Align weekly indicators to daily
    donchian_upper_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20w)
    donchian_lower_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20w)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_20w_aligned[i]) or np.isnan(donchian_lower_20w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper + above weekly EMA34 + volume spike
            if (close[i] > donchian_upper_20w_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_spike_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower + below weekly EMA34 + volume spike
            elif (close[i] < donchian_lower_20w_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_spike_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly Donchian lower or below weekly EMA34
            if close[i] < donchian_lower_20w_aligned[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly Donchian upper or above weekly EMA34
            if close[i] > donchian_upper_20w_aligned[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals