# NOTE: This strategy is being submitted as a complete rewrite in order to align with the new requirements (12h primary timeframe, HTF=1d). The hypothesis is that combining a Donchian breakout with 1-day trend confirmation and volume filters will yield fewer, higher-quality trades suitable for the 12h timeframe, with proper risk controls to avoid overtrading and ensure robustness across bull and bear markets.

#!/usr/bin/env python3
"""
12h_Donchian_20_Breakout_1D_Trend_VolumeFilter
Hypothesis: Donchian channel breakouts capture strong momentum. Filtering by 1-day trend (price > EMA50) and volume spikes avoids false signals in chop. Works in bull markets by riding breakouts and in bear markets by avoiding counter-trend entries. Target: 20-50 trades over 4 years.
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
    
    # Calculate Donchian channels (20-period)
    def donchian_channel(high, low, window):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(window - 1, len(high)):
            upper[i] = np.max(high[i - window + 1:i + 1])
            lower[i] = np.min(low[i - window + 1:i + 1])
        return upper, lower
    
    upper, lower = donchian_channel(high, low, 20)
    
    # Volume confirmation: 20-period moving average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20, 50)  # Donchian, volume MA20, EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume filter + 1d uptrend
            if close[i] > upper[i] and volume_filter and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume filter + 1d downtrend
            elif close[i] < lower[i] and volume_filter and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_20_Breakout_1D_Trend_VolumeFilter"
timeframe = "12h"
leverage = 1.0