#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with weekly trend filter and volume confirmation.
- Weekly trend: price > weekly SMA50 for long, < weekly SMA50 for short.
- Entry: breakout above/below 20-period Donchian channel on 12h.
- Volume: current volume > 20-period average volume.
- Exit: opposite Donchian breakout or trend reversal.
- Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_sma = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_sma_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma)
    
    # === 12H DONCHIAN CHANNEL (20-period) ===
    # Highest high and lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(weekly_sma_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly SMA
        bull_trend = close[i] > weekly_sma_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            if close[i] < donchian_low[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            if close[i] > donchian_high[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic: Donchian breakout in direction of weekly trend
            if bull_trend:
                # In bull trend: long on break above Donchian high
                if high[i] > donchian_high[i] and close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear trend: short on break below Donchian low
                if low[i] < donchian_low[i] and close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals