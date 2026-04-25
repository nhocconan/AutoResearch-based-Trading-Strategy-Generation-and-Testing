#!/usr/bin/env python3
"""
1h EMA(21) Pullback with 4h EMA(50) Trend and 1d Volume Spike Filter
Hypothesis: In strong trends (4h EMA50), 1h EMA21 pullbacks with volume confirmation
offer high-probability entries. Works in bull (pullbacks in uptrend) and bear
(pullbacks in downtrend) by trading with the 4h trend. Session filter (08-20 UTC)
reduces noise. Targets 60-150 total trades over 4 years.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA50 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 4h close for trend
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume spike filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(
        window=20, min_periods=20
    ).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h EMA21 for pullback entries
    ema_21 = pd.Series(close).ewm(
        span=21, adjust=False, min_periods=21
    ).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(21, 50)
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(ema_21[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_4h_aligned[i]
        vol_ma_1d = vol_ma_20_1d_aligned[i]
        ema_fast = ema_21[i]
        
        # Volume confirmation: current 1h volume > 1.5 * 20-day average volume (scaled)
        # Scale daily volume to hourly approximate (divide by 6.67 for 24h->1h)
        vol_ma_1h_approx = vol_ma_1d / 6.67
        volume_confirm = curr_volume > 1.5 * vol_ma_1h_approx
        
        if position == 0:
            # Look for entry signals
            # Long: price > EMA21 (pullback in uptrend) AND price > EMA50_4h (uptrend) AND volume confirmation
            long_entry = (curr_close > ema_fast and 
                         curr_close > ema_trend and volume_confirm)
            # Short: price < EMA21 (pullback in downtrend) AND price < EMA50_4h (downtrend) AND volume confirmation
            short_entry = (curr_close < ema_fast and 
                          curr_close < ema_trend and volume_confirm)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below EMA21 OR trend changes (price < EMA50_4h)
            if (curr_close < ema_fast or curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above EMA21 OR trend changes (price > EMA50_4h)
            if (curr_close > ema_fast or curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA21_Pullback_4hEMA50_Trend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0