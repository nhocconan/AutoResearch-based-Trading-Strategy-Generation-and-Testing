#!/usr/bin/env python3
"""
12h_donchian_breakout_1w_trend_volume_v1
Hypothesis: On 12h timeframe, enter long when price breaks above 20-period Donchian upper band with above-average volume and 1-week EMA trend bullish, enter short when price breaks below 20-period Donchian lower band with above-average volume and 1-week EMA trend bearish. Exit when price crosses the 20-period Donchian midpoint. Uses volume and trend filters to avoid false breakouts. Designed for 12-37 trades/year to minimize fee dust while capturing trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    if len(high) < 20 or len(low) < 20:
        return np.zeros(n)
    
    # Upper band: highest high over 20 periods
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume moving average for confirmation (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1-week EMA for trend filter (21-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(close[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle
            if close[i] < donchian_middle[i] and close[i-1] >= donchian_middle[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle
            if close[i] > donchian_middle[i] and close[i-1] <= donchian_middle[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian upper with bullish 1w EMA trend
                if close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower with bearish 1w EMA trend
                elif close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and ema_21_1w_aligned[i] < ema_21_1w_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals