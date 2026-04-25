#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_1dVolumeSpike_v1
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and 1d volume spike confirmation. 
Donchian breakouts capture momentum, 12h EMA50 ensures trend alignment, and 1d volume spike (>2.0x 20-bar mean) confirms institutional participation. 
Works in bull markets via upside breakouts and in bear markets via downside breakouts. Targets 12-25 trades/year to minimize fee drag.
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
    
    # Get 12h data for HTF trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA50 on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate 20-bar volume mean on 1d
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate Donchian channels (20-bar) on 6h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian, EMA50, volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or 
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current 6h volume > 2.0x 1d volume MA (scaled to 6h)
            # Approximate 6h volume equivalent: 1d volume / 4 (since 1d = 4x 6h)
            vol_6h_equiv = vol_ma20_1d_aligned[i] / 4.0
            vol_confirm = volume[i] > 2.0 * vol_6h_equiv
            
            # Long: price breaks above Donchian upper band in uptrend (close > 12h EMA50) with volume confirmation
            # Short: price breaks below Donchian lower band in downtrend (close < 12h EMA50) with volume confirmation
            long_signal = (close[i] > highest_high_20[i]) and (close[i] > ema50_12h_aligned[i]) and vol_confirm
            short_signal = (close[i] < lowest_low_20[i]) and (close[i] < ema50_12h_aligned[i]) and vol_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Donchian midpoint (mean reversion) or trend weakens
            donchian_mid = (highest_high_20[i] + lowest_low_20[i]) / 2.0
            exit_signal = close[i] < donchian_mid
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian midpoint
            donchian_mid = (highest_high_20[i] + lowest_low_20[i]) / 2.0
            exit_signal = close[i] > donchian_mid
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0