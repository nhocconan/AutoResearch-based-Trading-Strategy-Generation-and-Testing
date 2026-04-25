#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA50_TrendFilter_v2
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
In bull markets: long when price breaks above 20-day high AND weekly EMA50 up.
In bear markets: short when price breaks below 20-day low AND weekly EMA50 down.
Volume confirmation (1.5x 20-day average volume) filters low-quality breakouts.
Position size: 0.25 to limit drawdown in volatile markets.
Target: 15-25 trades/year to stay well under 150-trade 1d hard max.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need enough bars for EMA50
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day Donchian channels on primary timeframe
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = weekly EMA50 rising)
        if i >= start_idx + 1:
            htf_1w_bullish = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            htf_1w_bearish = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
        else:
            htf_1w_bullish = False
            htf_1w_bearish = False
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long setup: price breaks above 20-day high + weekly uptrend + volume
            long_setup = (close[i] > high_20[i]) and htf_1w_bullish and volume_confirmed
            
            # Short setup: price breaks below 20-day low + weekly downtrend + volume
            short_setup = (close[i] < low_20[i]) and htf_1w_bearish and volume_confirmed
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below 20-day low OR weekly trend turns bearish
            if (close[i] < low_20[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above 20-day high OR weekly trend turns bullish
            if (close[i] > high_20[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_TrendFilter_v2"
timeframe = "1d"
leverage = 1.0