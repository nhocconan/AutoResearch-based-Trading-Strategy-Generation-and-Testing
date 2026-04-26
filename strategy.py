#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrendFilter_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1w trend filter (price > 1w EMA34 for long, < for short) and volume spike confirmation.
Enters long when price breaks above R3 AND close > 1w EMA34 AND volume > 1.5 * 20-period average volume.
Enters short when price breaks below S3 AND close < 1w EMA34 AND volume > 1.5 * 20-period average volume.
Exits on opposite Camarilla level touch (R3 for long, S3 for short) or when price re-enters the Camarilla H-L range.
Uses 1w EMA34 for higher timeframe trend alignment to avoid counter-trend trades.
Volume spike confirms institutional participation. Camarilla levels provide mathematically derived support/resistance.
Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
Works in bull/bear markets by trading with the 1w trend and using volume to filter false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # Need to resample to 1d first to get daily OHLC
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R3/S3 as breakout levels
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1w EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_r3_aligned[i]
        breakout_short = close[i] < camarilla_s3_aligned[i]
        
        # Re-entry conditions (price back inside Camarilla H-L range)
        # Calculate Camarilla H-L for the day: H5 = close + 1.0*(high-low), L5 = close - 1.0*(high-low)
        camarilla_h5 = close_1d + 1.0 * (high_1d - low_1d)
        camarilla_l5 = close_1d - 1.0 * (high_1d - low_1d)
        camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
        camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
        price_in_range = (close[i] > camarilla_l5_aligned[i]) and (close[i] < camarilla_h5_aligned[i])
        
        if position == 0:
            # Long: breakout above R3 AND close > 1w EMA34 AND volume spike
            if breakout_long and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S3 AND close < 1w EMA34 AND volume spike
            elif breakout_short and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: breakout below S3 OR price re-enters Camarilla H-L range
            if breakout_short or price_in_range:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout above R3 OR price re-enters Camarilla H-L range
            if breakout_long or price_in_range:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrendFilter_VolumeSpike"
timeframe = "12h"
leverage = 1.0