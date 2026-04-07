#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: On daily timeframe, enter long when price breaks above 20-day Donchian high with above-average volume and weekly trend bullish (price above 20-week EMA), enter short when price breaks below 20-day Donchian low with above-average volume and weekly trend bearish (price below 20-week EMA). Exit when price crosses the 20-day Donchian midpoint. Designed for 10-25 trades/year to minimize fee dust while capturing breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Calculate 20-day Donchian channels
    if len(high) < 20:
        return np.zeros(n)
    
    # Donchian high (20-period rolling max)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low (20-period rolling min)
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint (for exits)
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly trend using 1w data (EMA20 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or np.isnan(ema_20w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midpoint
            if close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midpoint
            if close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian high with weekly uptrend
                if close[i] > donch_high[i] and close[i] > ema_20w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low with weekly downtrend
                elif close[i] < donch_low[i] and close[i] < ema_20w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals