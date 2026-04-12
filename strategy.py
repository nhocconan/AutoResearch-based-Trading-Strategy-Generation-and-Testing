#!/usr/bin/env python3
"""
1d_1w_donchian_breakout_volume
1-day Donchian channel breakout with weekly trend filter and volume confirmation.
Long when price breaks above 20-day high with weekly uptrend and volume surge.
Short when price breaks below 20-day low with weekly downtrend and volume surge.
Exit on opposite breakout or volatility expansion.
Designed for low trade frequency (<25/year) to minimize fee drag.
Works in bull (breakouts) and bear (breakdowns) via weekly trend filter.
"""

name = "1d_1w_donchian_breakout_volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly trend: price above/below 21-week EMA
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_uptrend = close_1w > ema_21_1w
    weekly_downtrend = close_1w < ema_21_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    # Daily Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: break above Donchian high with weekly uptrend and volume
        if (close[i] > donch_high[i] and weekly_uptrend_aligned[i] and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: break below Donchian low with weekly downtrend and volume
        elif (close[i] < donch_low[i] and weekly_downtrend_aligned[i] and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit on opposite breakout
        elif position == 1 and close[i] < donch_low[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals