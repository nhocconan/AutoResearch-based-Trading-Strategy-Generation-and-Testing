#!/usr/bin/env python3
"""
6h Donchian(20) breakout + 12h EMA50 trend filter + volume spike confirmation
Hypothesis: Donchian breakouts on 6h chart capture medium-term momentum. 
Filtering by 12h EMA50 trend ensures trades align with higher-timeframe direction.
Volume spike confirms institutional participation. Designed for 6h timeframe targeting 12-37 trades/year.
Uses discrete position sizing (0.25) to minimize fee churn. Works in bull markets via breakout continuation
and in bear markets via mean-reversion when price reverts to 12h EMA50 after overextended moves.
Uses proper MTF loading with get_htf_data called once before loop.
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
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_12h, Donchian, and volume MA to propagate
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema50_12h = ema_50_12h_aligned[i]
        upper_dc = donchian_high[i]
        lower_dc = donchian_low[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian AND uptrend (close > 12h EMA50) AND volume spike
            long_condition = (curr_close > upper_dc) and (curr_close > ema50_12h) and volume_spike
            # Short: price breaks below lower Donchian AND downtrend (close < 12h EMA50) AND volume spike
            short_condition = (curr_close < lower_dc) and (curr_close < ema50_12h) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price breaks below lower Donchian (reversal signal)
            if curr_close < lower_dc:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (reversal signal)
            if curr_close > upper_dc:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_breakout_12hEMA50_trend_volume_spike_v1"
timeframe = "6h"
leverage = 1.0