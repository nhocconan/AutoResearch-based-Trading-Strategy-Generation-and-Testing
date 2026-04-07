#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with 1-Day Supertrend Filter and Volume Confirmation
# Hypothesis: Breakouts from Donchian Channel (20) aligned with daily Supertrend direction
# and confirmed by volume spikes capture strong momentum moves while avoiding whipsaws.
# Works in bull markets (follow uptrend) and bear markets (follow downtrend) by using
# higher-timeframe trend filter. Target: 20-50 trades/year (80-200 total over 4 years).

name = "4h_donchian_breakout_1d_supertrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Donchian Channel (20) on 4h
    dc_period = 20
    upper_dc = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    lower_dc = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Supertrend on daily (ATR=10, multiplier=3.0)
    st_period = 10
    st_multiplier = 3.0
    
    # True Range for daily
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr_daily = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_daily = pd.Series(tr_daily).rolling(window=st_period, min_periods=st_period).mean().values
    
    # Supertrend calculation
    hl2_daily = (high_daily + low_daily) / 2
    upper_band = hl2_daily + st_multiplier * atr_daily
    lower_band = hl2_daily - st_multiplier * atr_daily
    
    # Initialize Supertrend arrays
    st_upper = np.full_like(close_daily, np.nan)
    st_lower = np.full_like(close_daily, np.nan)
    st_direction = np.full_like(close_daily, np.nan)  # 1=uptrend, -1=downtrend
    
    for i in range(st_period, len(close_daily)):
        if i == st_period:
            st_upper[i] = upper_band[i]
            st_lower[i] = lower_band[i]
        else:
            st_upper[i] = upper_band[i] if upper_band[i] < st_upper[i-1] or close_daily[i-1] > st_upper[i-1] else st_upper[i-1]
            st_lower[i] = lower_band[i] if lower_band[i] > st_lower[i-1] or close_daily[i-1] < st_lower[i-1] else st_lower[i-1]
        
        if i == st_period:
            st_direction[i] = 1 if close_daily[i] > st_upper[i] else -1
        else:
            if st_direction[i-1] == -1 and close_daily[i] > st_upper[i]:
                st_direction[i] = 1
            elif st_direction[i-1] == 1 and close_daily[i] < st_lower[i]:
                st_direction[i] = -1
            else:
                st_direction[i] = st_direction[i-1]
    
    # Align Supertrend direction to 4h timeframe
    st_direction_aligned = align_htf_to_ltf(prices, df_daily, st_direction)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(dc_period, st_period), n):
        # Skip if required data not available
        if (np.isnan(upper_dc[i]) or np.isnan(lower_dc[i]) or 
            np.isnan(st_direction_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian or trend changes to down
            if close[i] < lower_dc[i] or st_direction_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian or trend changes to up
            if close[i] > upper_dc[i] or st_direction_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above upper Donchian with uptrend
                if close[i] > upper_dc[i] and st_direction_aligned[i] == 1:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower Donchian with downtrend
                elif close[i] < lower_dc[i] and st_direction_aligned[i] == -1:
                    position = -1
                    signals[i] = -0.25
    
    return signals