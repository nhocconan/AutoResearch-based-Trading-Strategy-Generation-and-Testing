#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakout with 1d trend filter (price > 1d EMA50) and volume confirmation (>1.8x 20-period average) captures high-probability breakouts in both bull and bear markets. Added tighter volume threshold (2.0x) and require EMA50 slope > 0 for uptrend, < 0 for downtrend to reduce false signals. Target: 50-150 total trades over 4 years.
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
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # need for EMA50 + slope
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter with slope
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_slope = pd.Series(ema_50_1d).diff().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_slope)
    
    # Get 12h data for Camarilla calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for 12h
    # Camarilla levels based on previous bar's range
    # We need previous bar's OHLC for current bar's levels
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    # Set first value to NaN as there's no previous bar
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r1 = prev_close + 0.5 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 0.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Volume filter: volume > 2.0x 20-period average (tighter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(55, 20)  # EMA50 needs 50 periods + 5 for slope, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_50_1d_slope_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get aligned values
        ema_50_val = ema_50_1d_aligned[i]
        ema_50_slope = ema_50_1d_slope_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        
        # Get 12h close aligned for direct comparison
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        close_12h_val = close_12h_aligned[i]
        is_uptrend = (close_12h_val > ema_50_val) and (ema_50_slope > 0)
        is_downtrend = (close_12h_val < ema_50_val) and (ema_50_slope < 0)
        
        if position == 0:
            # Look for entry signals
            if is_uptrend:
                # Long conditions: price breaks above R1, volume spike
                long_signal = (close_12h_val > r1_val) and vol_spike[i]
            elif is_downtrend:
                # Short conditions: price breaks below S1, volume spike
                short_signal = (close_12h_val < s1_val) and vol_spike[i]
            else:
                long_signal = False
                short_signal = False
            
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
            # Exit conditions:
            # 1. Price closes below S1 (opposite Camarilla level)
            # 2. Price closes below pivot point (PP)
            if close_12h_val < s1_val or close_12h_val < pp_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price closes above R1 (opposite Camarilla level)
            # 2. Price closes above pivot point (PP)
            if close_12h_val > r1_val or close_12h_val > pp_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0