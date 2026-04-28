#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeTrend
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian upper channel (20-period high) with volume > 1.5x 20-period average volume, and short when price breaks below Donchian lower channel with volume confirmation. Use 1d EMA50 trend filter to align with higher timeframe trend, reducing counter-trend trades. Exit on opposite Donchian breakout with volume confirmation. This structure captures breakouts with institutional participation (volume) while filtering with 1d trend, aiming for low-moderate trade frequency (target: 20-50 trades/year) to minimize fee drag and work in bull/bear markets via trend alignment.
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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 4h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Daily trend: bullish when price > EMA50, bearish when price < EMA50
    daily_uptrend = close > ema50_daily_aligned
    daily_downtrend = close < ema50_daily_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for Donchian and EMA50 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_surge[i]) or np.isnan(ema50_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with daily trend alignment and volume surge
        long_entry = close[i] > donchian_high[i] and daily_uptrend[i] and volume_surge[i]
        short_entry = close[i] < donchian_low[i] and daily_downtrend[i] and volume_surge[i]
        
        # Exit on opposite Donchian breakout with volume surge (to avoid whipsaw)
        long_exit = close[i] < donchian_low[i] and volume_surge[i]
        short_exit = close[i] > donchian_high[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0