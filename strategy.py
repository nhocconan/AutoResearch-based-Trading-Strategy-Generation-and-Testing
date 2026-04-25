#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA50_TrendFilter_v1
Hypothesis: Trade daily Donchian(20) breakouts with weekly EMA50 trend filter.
Long when price breaks above 20-day high and weekly EMA50 rising.
Short when price breaks below 20-day low and weekly EMA50 falling.
Use volume confirmation to filter false breakouts.
Position size: 0.25 to limit drawdown.
Target: 10-20 trades/year to stay well under 150-trade 1d hard max.
Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (primary timeframe is 1d, so this is just the prices themselves)
    # But we need to calculate Donchian on 1d data - since prices is already 1d, we can use it directly
    # However, we need to ensure we're using completed daily bars for the Donchian calculation
    # For 1d timeframe, we can calculate Donchian directly from the prices dataframe
    
    # Calculate 20-day Donchian channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate weekly EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly EMA50 slope for trend direction (rising/falling)
    ema_50_1w_slope = np.diff(ema_50_1w, prepend=ema_50_1w[0])
    ema_50_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_slope)
    
    # Volume confirmation: current volume > 1.5 * 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and EMA50 (50)
    start_idx = max(lookback, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_50_1w_slope_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend: rising EMA50 = bullish, falling EMA50 = bearish
        htf_1w_bullish = ema_50_1w_slope_aligned[i] > 0
        htf_1w_bearish = ema_50_1w_slope_aligned[i] < 0
        
        if position == 0:
            # Long setup: price breaks above 20-day high + weekly uptrend + volume confirmation
            long_setup = (close[i] > highest_high[i]) and htf_1w_bullish and volume_confirmation[i]
            
            # Short setup: price breaks below 20-day low + weekly downtrend + volume confirmation
            short_setup = (close[i] < lowest_low[i]) and htf_1w_bearish and volume_confirmation[i]
            
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
            # Exit: price touches 20-day low OR weekly trend turns bearish
            if (close[i] <= lowest_low[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches 20-day high OR weekly trend turns bullish
            if (close[i] >= highest_high[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0