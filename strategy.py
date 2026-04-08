#!/usr/bin/env python3
"""
1d_1w_donchian_breakout_volume_v3
Hypothesis: Daily Donchian(20) breakout with weekly EMA(50) trend filter and volume confirmation.
Long when price breaks above Donchian upper band and weekly EMA > price.
Short when price breaks below Donchian lower band and weekly EMA < price.
Designed for low-frequency trading to minimize fee drag and work in both bull/bear markets.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for EMA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper band: highest high of last 20 periods
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low of last 20 periods
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower or trend breaks
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper or trend breaks
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper with volume and uptrend
            if close[i] > donchian_upper_aligned[i] and close[i] > ema_50_1w_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower with volume and downtrend
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema_50_1w_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals