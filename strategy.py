#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v2
Hypothesis: 4-hour Donchian breakout with 1-day trend filter and volume confirmation.
Long: price breaks above 20-period Donchian high with price above 1-day EMA50 and volume above average.
Short: price breaks below 20-period Donchian low with price below 1-day EMA50 and volume above average.
Exit: price crosses Donchian midline or trend reverses.
Designed for 25-40 trades/year on 4h timeframe with trend-following logic that works in both bull and bear markets.
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
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Price relative to Donchian levels
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        above_mid = close[i] > donchian_mid[i]
        below_mid = close[i] < donchian_mid[i]
        
        # 1d trend filter
        above_1d_ema50 = close[i] > ema50_1d_aligned[i]
        below_1d_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below midline or trend turns bearish
            if below_mid or below_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midline or trend turns bullish
            if above_mid or above_1d_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above Donchian high with volume confirmation and bullish trend
            if breakout_up and vol_confirmed and above_1d_ema50:
                position = 1
                signals[i] = 0.25
            # Short: breakout below Donchian low with volume confirmation and bearish trend
            elif breakout_down and vol_confirmed and below_1d_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals