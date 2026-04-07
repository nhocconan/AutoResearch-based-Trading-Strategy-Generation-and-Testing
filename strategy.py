#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 12h timeframe, use 1-day Donchian breakout for entry with volume confirmation and 1-day EMA trend filter. Enter long when price breaks above 1-day Donchian high with volume > 2x average and price above EMA; enter short when price breaks below 1-day Donchian low with volume > 2x average and price below EMA. Exit when price returns to the Donchian midpoint. Uses 1-day trend filter to work in both bull and bear markets, with volume confirmation to reduce false signals. Targets 15-30 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
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
    
    # 1-day data for Donchian channels, EMA trend filter, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian high and low (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1-day EMA for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # 1-day volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    donch_high_12h = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_12h = align_htf_to_ltf(prices, df_1d, donch_mid)
    ema_1d_12h = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or
            np.isnan(donch_mid_12h[i]) or np.isnan(ema_1d_12h[i]) or
            np.isnan(vol_ma_1d_12h[i]) or vol_ma_1d_12h[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 1-day average volume
        vol_confirm = volume[i] > 2.0 * vol_ma_1d_12h[i]
        
        # Trend direction from 1-day EMA
        uptrend = close[i] > ema_1d_12h[i]
        downtrend = close[i] < ema_1d_12h[i]
        
        if position == 1:  # Long position
            # Exit when price returns to Donchian midpoint
            if close[i] <= donch_mid_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price returns to Donchian midpoint
            if close[i] >= donch_mid_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume confirmation and uptrend
            long_entry = False
            if high[i] > donch_high_12h[i] and vol_confirm and uptrend:
                long_entry = True
            
            # Short entry: price breaks below Donchian low with volume confirmation and downtrend
            short_entry = False
            if low[i] < donch_low_12h[i] and vol_confirm and downtrend:
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals