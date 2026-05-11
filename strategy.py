#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v3
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) for breakout entries with volume confirmation and 1-day ADX trend filter.
In bull markets, breakouts from pivot levels capture momentum; in bear markets, reversals at S1/R1 during low volatility provide mean reversion.
Target: 25-35 trades/year to minimize fee drag while capturing strong moves with confluence.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla Pivot Levels (R1, S1) ---
    # Calculate pivot point and support/resistance levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)  # R1 = PP + (H-L)*1.1/12
    s1 = pivot - (range_1d * 1.1 / 12)  # S1 = PP - (H-L)*1.1/12
    
    # Align Camarilla levels to 4h timeframe (no extra delay needed as they're based on completed daily bar)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- 1d ADX for trend filter ---
    # Calculate True Range
    tr1 = pd.Series(high_1d).subtract(pd.Series(low_1d)).abs()
    tr2 = pd.Series(high_1d).subtract(pd.Series(close_1d).shift(1)).abs()
    tr3 = pd.Series(low_1d).subtract(pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    dm_plus = pd.Series(high_1d).diff()
    dm_minus = pd.Series(low_1d).diff().abs()
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
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    # --- Volume Spike Detection ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (1.8 * vol_ma.values)  # Volume spike threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or 
            np.isnan(s1_4h[i]) or
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
        long_breakout = (high[i] > r1_4h[i]) and vol_spike[i]
        short_breakout = (low[i] < s1_4h[i]) and vol_spike[i]
        
        # Mean reversion signals at Camarilla levels (only in ranging markets)
        long_reversion = (low[i] <= s1_4h[i]) and is_ranging  # Touch or penetrate S1
        short_reversion = (high[i] >= r1_4h[i]) and is_ranging  # Touch or penetrate R1
        
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
                # Exit long: price touches S1 or ADX drops (trend weakening)
                exit_signal = (low[i] <= s1_4h[i]) or (adx < 20)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches R1 or ADX drops
                exit_signal = (high[i] >= r1_4h[i]) or (adx < 20)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals