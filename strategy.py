#!/usr/bin/env python3
"""
4h_donchian_breakout_12h_trend_volume_v2
Hypothesis: 4-hour Donchian breakout with 12-hour EMA trend filter and volume confirmation.
Buy when price breaks above Donchian upper band with volume surge and price above 12h EMA50.
Sell when price breaks below Donchian lower band with volume surge and price below 12h EMA50.
Designed to capture trend continuation with momentum confirmation while avoiding false breakouts.
Works in bull markets (buying strength) and bear markets (selling weakness).
Target: 20-30 trades/year (80-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v2"
timeframe = "4h"
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
    
    # Donchian channel (20-period) on 4h
    donchian_window = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_window, n):
        # Skip if data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average (at least 1.5x)
        vol_confirmed = volume[i] > vol_ma[i] * 1.5
        
        # Price breakout conditions
        breakout_up = close[i] > donchian_upper[i]
        breakout_down = close[i] < donchian_lower[i]
        
        # 12h trend filter
        above_12h_ema50 = close[i] > ema50_12h_aligned[i]
        below_12h_ema50 = close[i] < ema50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower or trend turns bearish
            if close[i] < donchian_lower[i] or below_12h_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper or trend turns bullish
            if close[i] > donchian_upper[i] or above_12h_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above upper band with volume confirmation and bullish trend
            if breakout_up and vol_confirmed and above_12h_ema50:
                position = 1
                signals[i] = 0.25
            # Short: breakout below lower band with volume confirmation and bearish trend
            elif breakout_down and vol_confirmed and below_12h_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals