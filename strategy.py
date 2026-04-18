#!/usr/bin/env python3
"""
1d_PriceChannel_Breakout_VolumeRegime_v1
Strategy: 1d price channel breakout (Donchian) with weekly trend filter and volume confirmation.
Long: Close > 20-day high + weekly uptrend + volume spike
Short: Close < 20-day low + weekly downtrend + volume spike
Designed for 1d timeframe: ~10-20 trades/year per symbol (40-80 total over 4 years).
Uses volume regime filter to avoid chop and focus on high-conviction breakouts.
Works in bull/bear via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for 20-day high/low
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 20-day high and low (Donchian channels)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all data to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for weekly EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Volume confirmation (spike > 2x average)
        vol_confirm = volume[i] > 2.0 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > high_20_aligned[i]
        breakout_short = close[i] < low_20_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume spike + breakout above 20-day high
            if uptrend and vol_confirm and breakout_long:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume spike + breakout below 20-day low
            elif downtrend and vol_confirm and breakout_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or breakdown below 20-day low
            if not uptrend or close[i] < low_20_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or breakout above 20-day high
            if not downtrend or close[i] > high_20_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_PriceChannel_Breakout_VolumeRegime_v1"
timeframe = "1d"
leverage = 1.0