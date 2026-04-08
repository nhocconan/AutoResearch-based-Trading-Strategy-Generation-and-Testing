#!/usr/bin/env python3

# 1d_1w_trend_follow_v1
# Hypothesis: Weekly trend following with daily entries. Uses weekly Donchian channels for trend direction,
# daily price action for entry timing, and volume confirmation to avoid false breakouts.
# Designed to capture major trends while minimizing whipsaws in ranging markets.
# Target: 10-20 trades/year for low fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_trend_follow_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter - load once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_max = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1w, high_max)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_min)
    
    # Daily indicators
    # 20-period EMA for dynamic support/resistance
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema20[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > donchian_high[i]
        weekly_downtrend = close[i] < donchian_low[i]
        
        if position == 1:  # Long position
            # Exit: close below weekly Donchian low or EMA20
            if close[i] < donchian_low[i] or close[i] < ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above weekly Donchian high or EMA20
            if close[i] > donchian_high[i] or close[i] > ema20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Entry conditions
            if volume_ok:
                # Long entry: price breaks above weekly Donchian high in uptrend
                if weekly_uptrend and close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below weekly Donchian low in downtrend
                elif weekly_downtrend and close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals