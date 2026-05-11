#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Uses Camarilla pivot points from daily timeframe for S1/R1 levels with volume confirmation and 1-day ADX trend filter.
Trades breakouts in trending markets and mean reversion at pivot levels in ranging markets.
Target: 15-30 trades/year to minimize fee drag while capturing strong moves in both bull and bear markets.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla Pivot Points (S1, R1) ---
    # Calculate pivot point and levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # --- 1d ADX for trend filter (trending >25, ranging <20) ---
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).subtract(df_1d['low']).abs()
    tr2 = pd.Series(df_1d['high']).subtract(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).subtract(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = pd.Series(df_1d['high']).diff()
    dm_minus = pd.Series(df_1d['low']).diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # Calculate DX and ADX
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx_1d = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_1d_values = adx_1d.values
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    # --- Volume Spike Detection ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Significant volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
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
        is_ranging = adx < 20
        
        # Breakout signals at Camarilla levels
        long_breakout = (high[i] > r1_1d_aligned[i]) and vol_spike[i]
        short_breakout = (low[i] < s1_1d_aligned[i]) and vol_spike[i]
        
        # Mean reversion signals (only in ranging markets)
        long_reversion = (close[i] < s1_1d_aligned[i]) and is_ranging
        short_reversion = (close[i] > r1_1d_aligned[i]) and is_ranging
        
        if position == 0:
            if is_trending:
                # In trending markets, only take breakout signals
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout:
                    signals[i] = -0.25
                    position = -1
            else:
                # In ranging markets, take both breakout and mean reversion
                if long_breakout or long_reversion:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout or short_reversion:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches opposite Camarilla level or ADX drops (trend weakening)
                exit_signal = (low[i] < s1_1d_aligned[i]) or (adx < 20)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches opposite Camarilla level or ADX drops
                exit_signal = (high[i] > r1_1d_aligned[i]) or (adx < 20)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals