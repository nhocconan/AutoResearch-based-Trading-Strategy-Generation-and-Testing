#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA50 Trend Filter and Volume Spike Filter
Hypothesis: Weekly trend (1w EMA50) provides strong directional bias for 1d Donchian(20) breakouts.
Volume spike confirms momentum. Works in bull/bear by following weekly trend.
Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian(20) channels from prior day OHLC (avoid look-ahead)
    prev_high = prices['high'].shift(1).rolling(window=20, min_periods=20).max().values
    prev_low = prices['low'].shift(1).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for weekly EMA, 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(prev_high[i]) or np.isnan(prev_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        upper_channel = prev_high[i]
        lower_channel = prev_low[i]
        ema_50_level = ema_50_1w_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average volume
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Donchian breakout conditions
        broke_above_upper = curr_close > upper_channel
        broke_below_lower = curr_close < lower_channel
        
        # Trend alignment conditions
        above_ema = curr_close > ema_50_level
        below_ema = curr_close < ema_50_level
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close breaks below lower channel
            if position == 1:
                if curr_close < lower_channel:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above upper channel
            elif position == -1:
                if curr_close > upper_channel:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above upper channel AND above weekly EMA50 AND volume spike
            long_condition = broke_above_upper and above_ema and volume_spike
            
            # Short: break below lower channel AND below weekly EMA50 AND volume spike
            short_condition = broke_below_lower and below_ema and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0