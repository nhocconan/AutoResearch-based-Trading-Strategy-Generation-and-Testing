#!/usr/bin/env python3
# 4h_Donchian20_VolumeSurge_Trend_Filter
# Hypothesis: 4h Donchian(20) breakouts capture medium-term trends. Volume surge (>2x 20-period average) confirms institutional participation. 
# Trend filter: price above/below 50-period EMA ensures trading in direction of intermediate trend. 
# Works in bull markets (buy breakouts above upper band) and bear markets (sell breakdowns below lower band). 
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee-efficient range.

name = "4h_Donchian20_VolumeSurge_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 50-period EMA for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume surge: >2x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # wait for EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above upper Donchian band with volume surge and price above EMA50
            if (close[i] > donchian_high[i] and volume_surge[i] and close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian band with volume surge and price below EMA50
            elif (close[i] < donchian_low[i] and volume_surge[i] and close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian band (trend reversal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian band (trend reversal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals