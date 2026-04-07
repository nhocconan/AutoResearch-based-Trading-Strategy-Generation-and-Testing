#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + weekly EMA200 trend filter + volume confirmation.
Long when price breaks above 20-day high with weekly uptrend and volume surge.
Short when price breaks below 20-day low with weekly downtrend and volume surge.
Uses 1d timeframe for entry/exit and 1w for trend filter.
Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === DAILY DONCHIAN CHANNEL (20-day) ===
    # Use pandas rolling for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR weekly trend turns bearish
            if close[i] < donchian_low[i] or close[i] <= weekly_ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR weekly trend turns bullish
            if close[i] > donchian_high[i] or close[i] >= weekly_ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on weekly trend
            if close[i] > weekly_ema_aligned[i]:  # Weekly uptrend
                if close[i] > donchian_high[i]:  # Break above 20-day high
                    position = 1
                    signals[i] = 0.25
            else:  # Weekly downtrend
                if close[i] < donchian_low[i]:  # Break below 20-day low
                    position = -1
                    signals[i] = -0.25
    
    return signals