#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_volume_filter_v1
Hypothesis: Donchian channel breakouts on 12-hour timeframe filtered by 1-day volume confirmation and 1-day trend filter.
Long when price breaks above 20-period Donchian high with above-average volume and price above 1-day EMA50.
Short when price breaks below 20-period Donchian low with above-average volume and price below 1-day EMA50.
Designed for 20-30 trades/year on 12h timeframe with clear breakout logic that works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_volume_filter_v1"
timezone = "12h"
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
    
    # 1d data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Volume confirmation: 20-period average on 1d
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Trend filter: 1-day EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channel on 12h timeframe (20-period)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_period, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or vol_ma_1d_aligned[i] == 0 or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above 1-day average
        vol_confirmed = volume[i] > vol_ma_1d_aligned[i]
        
        # Trend filter
        above_1d_ema50 = close[i] > ema50_1d_aligned[i]
        below_1d_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend turns bearish
            if close[i] < donchian_low[i] or below_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend turns bullish
            if close[i] > donchian_high[i] or above_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume confirmation and bullish trend
            if close[i] > donchian_high[i] and vol_confirmed and above_1d_ema50:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume confirmation and bearish trend
            elif close[i] < donchian_low[i] and vol_confirmed and below_1d_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals