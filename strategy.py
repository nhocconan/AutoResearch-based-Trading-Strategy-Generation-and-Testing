#!/usr/bin/env python3
"""
4h_Donchian_20_Volume_Spike_ADX_Trend_Filter_v1
Hypothesis: Uses Donchian channel breakout (20-period) with volume spike confirmation and ADX trend filter (1-day).
In trending markets (ADX>25), takes breakout signals; in ranging markets (ADX<20), avoids trades to reduce whipsaw.
Volume spike ensures institutional participation. Designed for fewer, higher-quality trades to minimize fee drag.
"""

name = "4h_Donchian_20_Volume_Spike_ADX_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d ADX for trend filter ---
    # True Range
    tr1 = pd.Series(df_1d['high']).subtract(df_1d['low']).abs()
    tr2 = pd.Series(df_1d['high']).subtract(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).subtract(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(df_1d['high']).diff()
    dm_minus = pd.Series(df_1d['low']).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Wilder's smoothing (alpha = 1/14)
    atr_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx_1d = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d_values = adx_1d.values
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    # --- Donchian Channel (20-period) ---
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Spike Detection ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Significant volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime based on ADX
        adx = adx_1d_aligned[i]
        is_trending = adx > 25
        
        # Breakout signals
        long_breakout = (high[i] > highest_20[i]) and vol_spike[i]
        short_breakout = (low[i] < lowest_20[i]) and vol_spike[i]
        
        if position == 0:
            # Only take breakout signals in trending markets
            if is_trending:
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches opposite Donchian band or ADX drops below 20 (trend weakening)
                exit_signal = (low[i] < lowest_20[i]) or (adx < 20)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches opposite Donchian band or ADX drops below 20
                exit_signal = (high[i] > highest_20[i]) or (adx < 20)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals