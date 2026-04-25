#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 12h EMA50 Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
Filtered by 12h EMA50 trend and volume spike to avoid false breakouts.
Works in both bull and bear markets by taking breakouts in direction of higher timeframe trend.
Designed for 4h timeframe targeting 20-50 trades/year.
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
    
    # Get 12h data for EMA (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) on 4h: upper = max(high, 20), lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_close > upper) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Donchian lower AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_close < lower) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below Donchian lower OR price crosses below EMA
            if (curr_close < lower) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price crosses above Donchian upper OR price crosses above EMA
            if (curr_close > upper) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0