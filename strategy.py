#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: Uses 12h Donchian breakouts with 1d trend filter and volume confirmation.
# Enters long when price breaks above Donchian upper (20) with volume spike and daily uptrend.
# Enters short when price breaks below Donchian lower (20) with volume spike and daily downtrend.
# Exits on opposite break or trend failure. Designed for 12-37 trades/year to avoid fee drag.
# Uses 1d trend filter for multi-timeframe alignment and Donchian channels as structure.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h volume average (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: breakdown below Donchian lower or daily trend failure
            if close[i] < donch_low[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: break above Donchian upper or daily trend failure
            if close[i] > donch_high[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: above Donchian upper with volume spike and daily uptrend
                if close[i] > donch_high[i] and daily_uptrend:
                    position = 1
                    signals[i] = 0.25
                # Short entry: below Donchian lower with volume spike and daily downtrend
                elif close[i] < donch_low[i] and daily_downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals