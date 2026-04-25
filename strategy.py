#!/usr/bin/env python3
"""
12h Donchian(20) Breakout with 1d EMA34 Trend Filter and Volume Spike
Hypothesis: Donchian breakouts capture momentum shifts. Using 1d EMA34 as higher-timeframe trend filter ensures alignment with daily trend, reducing false signals. Volume spike confirms breakout strength. Works in bull markets (breakouts above upper channel) and bear markets (breakdowns below lower channel) by requiring trend alignment. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-37 trades/year on 12h.
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
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian Channel (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 1.5 * 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian calculations
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_channel = period20_high[i]
        lower_channel = period20_low[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper channel AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (curr_close > upper_channel) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below lower channel AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (curr_close < lower_channel) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below lower channel OR price < 1d EMA34 (trend change)
            if (curr_close < lower_channel) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper channel OR price > 1d EMA34 (trend change)
            if (curr_close > upper_channel) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_VolumeSpike_1dEMA34_Trend"
timeframe = "12h"
leverage = 1.0