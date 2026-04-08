#!/usr/bin/env python3
# daily_price_action_v1
# Hypothesis: Price action strategy on daily timeframe using weekly trend filter.
# Long when: daily close > weekly EMA200 and daily close breaks above daily Donchian upper (20)
# Short when: daily close < weekly EMA200 and daily close breaks below daily Donchian lower (20)
# Exit when: price crosses back below/above weekly EMA200 or daily Donchian middle
# Uses weekly trend to filter direction and daily breakout for entry timing.
# Target: 15-25 trades/year (~60-100 total over 4 years) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_price_action_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    donch_len = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=donch_len, min_periods=donch_len).max().values
    donch_low = low_series.rolling(window=donch_len, min_periods=donch_len).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_len = 200
    ema200_1w = pd.Series(close_1w).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_len, ema_len)
    
    for i in range(start_idx, n):
        # Skip if weekly EMA not available
        if np.isnan(ema200_1w_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Close below weekly EMA200 or below daily Donchian middle
            if close[i] < ema200_1w_aligned[i] or close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Close above weekly EMA200 or above daily Donchian middle
            if close[i] > ema200_1w_aligned[i] or close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Close above weekly EMA200 and breaks above daily Donchian upper
            if close[i] > ema200_1w_aligned[i] and close[i] > donch_high[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Close below weekly EMA200 and breaks below daily Donchian lower
            elif close[i] < ema200_1w_aligned[i] and close[i] < donch_low[i]:
                position = -1
                signals[i] = -0.25
    
    return signals