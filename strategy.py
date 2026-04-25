#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: On 1d timeframe, Camarilla R1/S1 breakout with 1w trend filter (price > 1w EMA50) and volume confirmation (>1.8x 20-period average) captures high-probability breakouts in both bull and bear markets. The Camarilla levels act as intraday support/resistance: long when price breaks above R1 in 1w uptrend; short when price breaks below S1 in 1w downtrend. Exits on opposite Camarilla level (S1 for longs, R1 for shorts) or close below/above the pivot point. Designed for ~50-80 total trades over 4 years (12-20/year) via tight entry conditions requiring multi-timeframe alignment.
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
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need for EMA50
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # Camarilla levels based on previous day's range
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.1*(high-low)
    # R2 = close + 0.7*(high-low)
    # R1 = close + 0.5*(high-low)
    # PP = (high+low+close)/3
    # S1 = close - 0.5*(high-low)
    # S2 = close - 0.7*(high-low)
    # S3 = close - 1.1*(high-low)
    # S4 = close - 1.5*(high-low)
    
    # We need previous day's OHLC for today's levels
    # Shift by 1 to get previous day's values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first value to NaN as there's no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r1 = prev_close + 0.5 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 0.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1d timeframe (they are already for 1d)
    # But we need to align them to the 1d timeframe for consistency
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20)  # EMA50 needs 50 periods, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get aligned values
        ema_50_val = ema_50_1w_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        
        # 1w trend filter: price vs EMA50
        # Get 1w close aligned for direct comparison
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        close_1w_val = close_1w_aligned[i]
        is_uptrend = close_1w_val > ema_50_val
        
        if position == 0:
            # Look for entry signals
            if is_uptrend:
                # Long conditions: price breaks above R1, volume spike
                long_signal = (close[i] > r1_val) and vol_spike[i]
            else:
                # Short conditions: price breaks below S1, volume spike
                short_signal = (close[i] < s1_val) and vol_spike[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price closes below S1 (opposite Camarilla level)
            # 2. Price closes below pivot point (PP)
            if close[i] < s1_val or close[i] < pp_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price closes above R1 (opposite Camarilla level)
            # 2. Price closes above pivot point (PP)
            if close[i] > r1_val or close[i] > pp_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0