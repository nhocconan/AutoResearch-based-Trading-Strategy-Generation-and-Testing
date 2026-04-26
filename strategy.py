#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: On 4h timeframe, Donchian channel (20-bar) breakouts with 12h trend filter (price > 12h EMA34 for long, < for short) and volume confirmation (>2x avg) provides robust directional signals. Works in bull markets (long when price > 12h EMA34 + upper Donchian breakout) and bear markets (short when price < 12h EMA34 + lower Donchian breakdown). Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 75-200 trades over 4 years (19-50/year) for optimal 4h frequency. 12h trend filter avoids whipsaws in counter-trend breakouts while volume spike confirms institutional participation.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian warmup (20) + EMA warmup (34) + volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average
        
        if position == 0:
            # Long: price > 12h EMA34 + breaks above upper Donchian + volume
            long_signal = (close[i] > ema_34_12h_aligned[i] and 
                          close[i] > donchian_upper[i] and 
                          vol_confirmed)
            
            # Short: price < 12h EMA34 + breaks below lower Donchian + volume
            short_signal = (close[i] < ema_34_12h_aligned[i] and 
                           close[i] < donchian_lower[i] and 
                           vol_confirmed)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below 12h EMA34 OR breaks below lower Donchian (reversal)
            if close[i] < ema_34_12h_aligned[i] or close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above 12h EMA34 OR breaks above upper Donchian (reversal)
            if close[i] > ema_34_12h_aligned[i] or close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0