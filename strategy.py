#!/usr/bin/env python3
"""
12h_donchian_breakout_1w_trend_volume_v1
Hypothesis: Donchian(20) breakout on 12h timeframe with 1-week EMA trend filter and volume confirmation.
Breakouts above/below 20-period Donchian channels signal trend continuation.
Weekly EMA filter ensures alignment with higher timeframe trend.
Volume confirmation adds conviction to breakouts.
Designed for 12-37 trades/year on 12h timeframe with strong trend-following edge that works in both bull and bear markets.
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
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Donchian channels (20-period) on 12h data
    # Use rolling window on 12h high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Trend filter: price relative to weekly EMA20
        above_weekly_ema = close[i] > ema20_1w_aligned[i]
        below_weekly_ema = close[i] < ema20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend turns bearish
            if close[i] < donchian_low[i] or below_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend turns bullish
            if close[i] > donchian_high[i] or above_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above Donchian high with volume confirmation and bullish trend
            if close[i] > donchian_high[i] and vol_confirmed and above_weekly_ema:
                position = 1
                signals[i] = 0.25
            # Short: breakdown below Donchian low with volume confirmation and bearish trend
            elif close[i] < donchian_low[i] and vol_confirmed and below_weekly_ema:
                position = -1
                signals[i] = -0.25
    
    return signals