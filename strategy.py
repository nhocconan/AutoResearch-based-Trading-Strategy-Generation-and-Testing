#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with 1d Trend Filter and Volume
# Hypothesis: Donchian(20) breakouts on 12h chart capture strong trends.
# Filtered by 1d EMA50 to ensure alignment with higher timeframe direction.
# Volume confirmation ensures breakout validity.
# Works in both bull and bear markets as trend filter adapts.
# Targets 15-25 trades/year to minimize fee drag.

name = "12h_donchian20_1d_ema_volume_v1"
timeframe = "12h"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=10).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=10).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend turns bearish
            if close[i] < donchian_low[i] or close[i] < ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend turns bullish
            if close[i] > donchian_high[i] or close[i] > ema50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout above Donchian high with volume and bullish trend
            if vol_spike[i] and close[i] > donchian_high[i] and (i == 50 or close[i-1] <= donchian_high[i-1]) and close[i] > ema50_12h[i]:
                position = 1
                signals[i] = 0.25
            # Breakout below Donchian low with volume and bearish trend
            elif vol_spike[i] and close[i] < donchian_low[i] and (i == 50 or close[i-1] >= donchian_low[i-1]) and close[i] < ema50_12h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals