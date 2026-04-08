#!/usr/bin/env python3
# 4h_kama_trend_volume_v1
# Hypothesis: Uses KAMA to determine trend direction and Donchian breakout with volume confirmation.
# Goes long when price breaks above Donchian high in KAMA uptrend with volume surge.
# Goes short when price breaks below Donchian low in KAMA downtrend with volume surge.
# Designed for low trade frequency (20-50/year) to avoid fee drift, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_kama_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA for trend direction
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = abs(close_series.diff(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * 0.58 + 0.42) ** 2
    kama = close_series.copy()
    for i in range(1, len(kama)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close_series.iloc[i] - kama.iloc[i-1])
    kama_values = kama.values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(kama_values[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price relative to KAMA
        kama_uptrend = close[i] > kama_values[i]
        kama_downtrend = close[i] < kama_values[i]
        
        # Donchian breakout signals
        breakout_high = close[i] > donchian_high[i-1]
        breakout_low = close[i] < donchian_low[i-1]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: Donchian breakdown or trend change
            if close[i] < donchian_low[i] or not kama_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian breakout or trend change
            if close[i] > donchian_high[i] or not kama_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: Donchian breakout in uptrend
                if kama_uptrend and breakout_high:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Donchian breakdown in downtrend
                elif kama_downtrend and breakout_low:
                    position = -1
                    signals[i] = -0.25
    
    return signals