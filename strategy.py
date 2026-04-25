#!/usr/bin/env python3
"""
6h Weekly Pivot + Donchian(20) Breakout + Volume Spike
Hypothesis: Weekly pivot levels (from prior week) act as major support/resistance. 
Breakouts above weekly R1 or below S1 with Donchian(20) confirmation and volume spike 
indicate institutional participation. Works in both bull/bear markets by only taking 
breakouts in direction of weekly trend (price vs weekly VWAP). 
Target: 12-37 trades/year on 6h (50-150 total over 4 years).
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
    
    # Get 1w data for weekly pivot calculation and trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly VWAP for trend filter (typical price * volume cumsum / volume cumsum)
    typical_price = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    vol = df_1w['volume'].values
    vp = typical_price * vol
    vwap_1w = np.nancumsum(vp) / np.nancumsum(vol)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Calculate weekly pivot points from previous 1w OHLC
    # Standard formula: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    prev_close = df_1w['close'].values
    prev_high = df_1w['high'].values
    prev_low = df_1w['low'].values
    weekly_pivot = (prev_high + prev_low + prev_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - prev_low
    weekly_s1 = 2 * weekly_pivot - prev_high
    
    # Align weekly levels to 6h (use previous week's levels for current week's trading)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate Donchian(20) breakout levels on 6h data
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly data alignment and Donchian
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vwap_1w_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        weekly_pivot_level = weekly_pivot_aligned[i]
        weekly_r1_level = weekly_r1_aligned[i]
        weekly_s1_level = weekly_s1_aligned[i]
        vwap_trend = vwap_1w_aligned[i]
        donchian_up = donchian_upper[i]
        donchian_low = donchian_lower[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above weekly R1 AND Donchian upper AND volume spike AND price > weekly VWAP (uptrend)
            long_entry = (curr_high > weekly_r1_level) and (curr_high > donchian_up) and vol_spike and (curr_close > vwap_trend)
            # Short: price breaks below weekly S1 AND Donchian lower AND volume spike AND price < weekly VWAP (downtrend)
            short_entry = (curr_low < weekly_s1_level) and (curr_low < donchian_low) and vol_spike and (curr_close < vwap_trend)
            
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
            # Exit: price crosses below weekly pivot (reversal) OR price < weekly VWAP (trend change)
            if (curr_close < weekly_pivot_level) or (curr_close < vwap_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above weekly pivot (reversal) OR price > weekly VWAP (trend change)
            if (curr_close > weekly_pivot_level) or (curr_close > vwap_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeSpike_1wVWAP_Trend"
timeframe = "6h"
leverage = 1.0