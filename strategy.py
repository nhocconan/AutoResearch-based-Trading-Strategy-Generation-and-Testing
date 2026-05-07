#!/usr/bin/env python3
"""
4H_Donchian_20_Volume_Trend_v1
Hypothesis: 4H Donchian channel breakout with volume confirmation and 1D EMA trend filter.
Works in bull/bear markets: Breakouts capture strong moves; volume filter ensures validity; EMA filter aligns with higher timeframe trend to avoid counter-trend trades.
Targets 20-50 trades/year to minimize fee drag on 4H timeframe.
"""
name = "4H_Donchian_20_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1D data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) on 4H data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1D EMA34 for trend direction
    close_1d_series = pd.Series(df_1d['close'])
    ema_34 = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current 4H volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 34)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 20 bars between trades (~10 days on 4H TF) to reduce frequency
            if bars_since_exit < 20:
                continue
                
            # Long: price breaks above Donchian high with EMA uptrend and volume spike
            if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and 
                close[i] > ema_34_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below Donchian low with EMA downtrend and volume spike
            elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and 
                  close[i] < ema_34_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Donchian level (mean reversion within channel)
            if position == 1 and close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals