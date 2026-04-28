#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeFilter
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) high with volume surge and 12h uptrend (EMA25 > EMA50), short when price breaks below Donchian(20) low with volume surge and 12h downtrend. Exit on opposite Donchian level break with volume. Trend filter from 12h EMA avoids counter-trend trades, volume surge confirms institutional participation, and Donchian breakouts capture breakout momentum. Designed for moderate trade frequency (~20-50/year) to balance signal quality and fee decay in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h 25 and 50 EMA for trend filter
    close_12h = df_12h['close'].values
    ema25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h trend: bullish when EMA25 > EMA50, bearish when EMA25 < EMA50
    trend_up = ema25_12h_aligned > ema50_12h_aligned
    trend_down = ema25_12h_aligned < ema50_12h_aligned
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema25_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with 12h trend alignment and volume surge
        long_entry = close[i] > donchian_high[i] and trend_up[i] and volume_surge[i]
        short_entry = close[i] < donchian_low[i] and trend_down[i] and volume_surge[i]
        
        # Exit on opposite Donchian level break with volume surge
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

name = "4h_Donchian20_Breakout_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0