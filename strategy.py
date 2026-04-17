#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h EMA trend filter + Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-period high with 12h EMA34 > EMA50 and volume > 1.5x 20-period volume average.
Short when price breaks below 20-period low with 12h EMA34 < EMA50 and volume > 1.5x 20-period volume average.
Uses discrete position sizing (0.25) to minimize fee churn. Designed to capture medium-term trends
with confirmation from higher timeframe trend and volume, reducing false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 and EMA50
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high_4h, low_4h, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (4h)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-period high with 12h EMA34 > EMA50 and volume
            if (close[i] > donchian_upper_aligned[i] and 
                ema34_12h_aligned[i] > ema50_12h_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with 12h EMA34 < EMA50 and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema34_12h_aligned[i] < ema50_12h_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-period low (opposite side of channel)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-period high (opposite side of channel)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hEMA34_50_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0