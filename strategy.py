#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d EMA trend filter and volume spike confirmation
# Designed for 20-40 trades/year with proper risk control via trend failure
# Long: price breaks above Donchian(20) high + price > 1d EMA50 + volume spike
# Short: price breaks below Donchian(20) low + price < 1d EMA50 + volume spike
# Exit: trend failure (price crosses 1d EMA50) or opposite breakout
# Volume filter: current 1d volume > 1.8x 20-day average to avoid false breakouts
# Works in bull markets via trend-following breakouts and in bear via mean-reversion at extremes

name = "4h_Donchian_Breakout_EMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-day average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1d volume > 1.8x 20-day average
        # Find the most recent completed 1d bar
        idx_1d = len(df_1d) - 1
        while idx_1d >= 0 and df_1d.iloc[idx_1d]['open_time'] > prices.iloc[i]['open_time']:
            idx_1d -= 1
        vol_filter = False
        if idx_1d >= 0:
            vol_1d_current = df_1d.iloc[idx_1d]['volume']
            vol_filter = vol_1d_current > 1.8 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for breakout with trend and volume confirmation
            # Long: price breaks above Donchian high + uptrend + volume spike
            if close[i] > donchian_high[i] and ema50_aligned[i] > 0:
                if vol_filter:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Donchian low + downtrend + volume spike
            elif close[i] < donchian_low[i] and ema50_aligned[i] < 0:
                if vol_filter:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: trend failure (price crosses below EMA50) or opposite breakout
            if ema50_aligned[i] <= 0 or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend failure (price crosses above EMA50) or opposite breakout
            if ema50_aligned[i] >= 0 or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals