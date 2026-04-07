#!/usr/bin/env python3
"""
4-hour Donchian Breakout with 1-day Trend Filter and Volume Confirmation
Hypothesis: Price breaking above/below 4-hour Donchian channels (20-period) with 
1-day EMA trend filter and volume confirmation provides robust entries in both 
bull and bear markets. The 1-day EMA determines trend bias (bullish when price > EMA50, 
bearish when price < EMA50), while volume above 20-period average confirms momentum.
Breakouts are filtered by trend to avoid counter-trend trades, and positions are 
exited when price reverses to the midpoint of the Donchian channel or trend changes.
Target: 20-50 trades per year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
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
    
    # === 1D TREND FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)  # already shifted
    
    # === 4H DONCHIAN CHANNELS (LTF) ===
    donchian_len = 20
    # Calculate rolling high/low for Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_low = low_series.rolling(window=donchian_len, min_periods=donchian_len).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_len, n):
        if np.isnan(one_d_ema_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        bull_trend = close[i] > one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian midpoint OR trend turns bearish
            if close[i] < donchian_mid[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian midpoint OR trend turns bullish
            if close[i] > donchian_mid[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on 1d trend and Donchian breakout
            if bull_trend:
                # In bull market: long on break above Donchian high
                if high[i] > donchian_high[i] and close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.30
            else:
                # In bear market: short on break below Donchian low
                if low[i] < donchian_low[i] and close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals