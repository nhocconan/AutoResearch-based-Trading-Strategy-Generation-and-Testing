#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
- Long when price breaks above 4h Donchian upper band AND price > 1d EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below 4h Donchian lower band AND price < 1d EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses the 4h Donchian midpoint (mean reversion to median)
- Uses 1d EMA34 for HTF trend alignment to avoid counter-trend trades
- Volume spike ensures institutional participation and reduces false breakouts
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    donchian_high_4h = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    
    # Align Donchian levels to 4h timeframe
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 35, 21)  # Need 20 for Donchian, 35 for EMA34 (34+1), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_4h_aligned[i]) or 
            np.isnan(donchian_low_4h_aligned[i]) or 
            np.isnan(donchian_mid_4h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions (using 4h Donchian channels)
        breakout_up = close[i] > donchian_high_4h_aligned[i]  # Break above Donchian upper band
        breakout_down = close[i] < donchian_low_4h_aligned[i]  # Break below Donchian lower band
        
        # Trend filter (using 1d EMA34)
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
            # Exit: price crosses 4h Donchian midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint
                if close[i] < donchian_mid_4h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above midpoint
                if close[i] > donchian_mid_4h_aligned[i]:
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