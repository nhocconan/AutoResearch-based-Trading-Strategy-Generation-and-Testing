#!/usr/bin/env python3
# 4h_donchian_20_1d_trend_volume_v3
# Hypothesis: 4-hour Donchian(20) breakout in direction of 1-day EMA(50) trend, confirmed by volume surge (>1.5x 20-period average).
# Long when price breaks above 20-period high and 1d EMA(50) > EMA(50) one period ago with volume surge.
# Short when price breaks below 20-period low and 1d EMA(50) < EMA(50) one period ago with volume surge.
# Uses volume confirmation to avoid false breakouts and reduce whipsaw.
# Designed for 20-40 trades/year on 4h to minimize fee drag while capturing trending moves.
# Works in bull markets via upward breakouts and bear markets via downward breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4-hour Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1-day EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA(50) and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band
            if close[i] < low_min_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band
            if close[i] > high_max_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper band with 1d uptrend and volume surge
            if close[i] > high_max_20[i] and ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band with 1d downtrend and volume surge
            elif close[i] < low_min_20[i] and ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals