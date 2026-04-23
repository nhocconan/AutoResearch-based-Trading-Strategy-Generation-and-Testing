#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
- Long when price breaks above 12h Donchian upper (20-period high) AND price > 1d EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below 12h Donchian lower (20-period low) AND price < 1d EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses the 12h Donchian midpoint (mean reversion to median)
- Uses 1d EMA34 for trend alignment to avoid counter-trend trades
- Volume spike ensures institutional participation and reduces false breakouts
- Donchian channels provide clear structural levels that work in both bull and bear markets
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Donchian upper = 20-period high, lower = 20-period low
    donchian_upper_12h = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_lower_12h = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid_12h = (donchian_upper_12h + donchian_lower_12h) / 2.0
    
    # Align 12h Donchian levels to 12h LTF (no additional delay needed for Donchian)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_12h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid_12h)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 35, 21)  # Need 20 for Donchian, 35 for EMA34 (34+1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_upper_aligned[i]  # Break above Donchian upper
        breakout_down = close[i] < donchian_lower_aligned[i]  # Break below Donchian lower
        
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
            # Exit: price crosses Donchian midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint
                if close[i] < donchian_mid_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above midpoint
                if close[i] > donchian_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0