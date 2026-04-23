#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
- Long when price breaks above 4h Donchian upper band AND price > 1d EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below 4h Donchian lower band AND price < 1d EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses the 4h Donchian middle band (mean reversion to median)
- Uses 1d EMA34 for trend alignment to avoid counter-trend trades and capture major trend
- Volume spike ensures institutional participation and reduces false breakouts
- Uses 4h primary timeframe with 1d HTF for signal direction to balance trade frequency
- Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 35)  # Need 20 for Donchian, 35 for EMA34 (34+1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_upper[i]  # Break above Donchian upper band
        breakout_down = close[i] < donchian_lower[i]  # Break below Donchian lower band
        
        # Trend filter
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses Donchian middle band (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle band
                if close[i] < donchian_middle[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above middle band
                if close[i] > donchian_middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0