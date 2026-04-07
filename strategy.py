#!/usr/bin/env python3
"""
12h_price_channel_breakout_1w_trend_volume_v1
Hypothesis: On 12h timeframe, enter long when price breaks above weekly Donchian high with volume above 20-period average and 1d close above 20 EMA, enter short when price breaks below weekly Donchian low with volume above average and 1d close below 20 EMA. Exit when price crosses the 20-period EMA. Uses 1d EMA trend filter to avoid counter-trend trades. Designed for 15-30 trades/year to minimize fee drag while capturing trend continuation in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_price_channel_breakout_1w_trend_volume_v1"
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
    
    # Calculate 20-period EMA for trend filter and exit
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly Donchian channels for breakout signals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian high (20-period)
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Weekly Donchian low (20-period)
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 12h timeframe
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Calculate 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema_20[i]) or np.isnan(vol_ma[i]) or np.isnan(close[i]) or 
            np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 20-period EMA
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 20-period EMA
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above weekly Donchian high with 1d EMA uptrend
                if close[i] > donchian_high_1w_aligned[i] and close[i-1] <= donchian_high_1w_aligned[i-1] and ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below weekly Donchian low with 1d EMA downtrend
                elif close[i] < donchian_low_1w_aligned[i] and close[i-1] >= donchian_low_1w_aligned[i-1] and ema_20_1d_aligned[i] < ema_20_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals