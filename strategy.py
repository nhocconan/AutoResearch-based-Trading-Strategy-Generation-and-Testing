#!/usr/bin/env python3
"""
1h Camarilla H3L3 Breakout + 4h EMA50 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels on 1d act as key support/resistance. Breakouts above H3 or below L3
on 1h capture momentum in the direction of the 4h EMA50 trend. Volume spike confirms participation.
Designed for 1h timeframe with tight entry conditions to achieve 15-37 trades/year (60-150 over 4 years).
Uses 4h for signal direction (EMA50 trend) and 1d for Camarilla pivot levels. Session filter (08-20 UTC)
reduces noise. Works in bull (breakouts above H3 in uptrend) and bear (breakouts below L3 in downtrend).
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
    open_time = prices['open_time'].values
    
    # Get 1d data for Camarilla pivot (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 1d OHLC (H3, L3, H4, L4)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    # H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    rng = high_1d - low_1d
    H3 = close_1d + 1.1 * rng * 1.1 / 4
    L3 = close_1d - 1.1 * rng * 1.1 / 4
    H4 = close_1d + 1.1 * rng * 1.1 / 2
    L4 = close_1d - 1.1 * rng * 1.1 / 2
    
    # Align Camarilla levels to 1h (no extra delay needed - levels based on completed 1d bar)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Get 4h data for EMA50 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Camarilla (1d), EMA, volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        H4_level = H4_aligned[i]
        L4_level = L4_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_high > H3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below L3 AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_low < L3_level) and vol_spike and (curr_close < ema_trend)
            
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
            # Exit: price crosses below L3 (breakdown) OR price crosses below EMA (trend change)
            if (curr_low < L3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above H3 (breakout) OR price crosses above EMA (trend change)
            if (curr_high > H3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0