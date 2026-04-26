#!/usr/bin/env python3
"""
12h_Camarilla_H4_H5_Breakout_1dTrend_ChaikinVolume_v1
Hypothesis: 12h Camarilla H4/H5 breakout with 1d trend filter and Chaikin Money Flow volume confirmation.
- Uses 12h timeframe for low trade frequency (target: 50-150 total trades over 4 years)
- Camarilla H4/H5 levels from 1d provide strong support/resistance from prior day
- 1d EMA50 filter ensures trades align with higher timeframe trend
- Chaikin Money Flow (20) > 0 confirms institutional buying/selling pressure
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the 1d trend and using CMF to filter false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Chaikin Money Flow (20) on 1d for volume confirmation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier
    mfm = np.where((high_1d - low_1d) != 0, ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d), 0.0)
    # Money Flow Volume
    mfv = mfm * volume_1d
    # CMF(20) = 20-period sum of MFV / 20-period sum of volume
    cmf_20 = np.full_like(close_1d, np.nan)
    for i in range(19, len(close_1d)):
        if not np.isnan(np.nansum(mfv[i-19:i+1])) and not np.isnan(np.nansum(volume_1d[i-19:i+1])):
            cmf_20[i] = np.nansum(mfv[i-19:i+1]) / np.nansum(volume_1d[i-19:i+1])
    cmf_20_aligned = align_htf_to_ltf(prices, df_1d, cmf_20)
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_h4 = close_1d + (1.0/6) * (high_1d - low_1d)  # H4 = close + 1/6*(high-low)
    camarilla_h5 = close_1d + (1.0/4) * (high_1d - low_1d)  # H5 = close + 1/4*(high-low)
    camarilla_l4 = close_1d - (1.0/6) * (high_1d - low_1d)  # L4 = close - 1/6*(high-low)
    camarilla_l5 = close_1d - (1.0/4) * (high_1d - low_1d)  # L5 = close - 1/4*(high-low)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 20 for CMF)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(cmf_20_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h5_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_l5_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_h5_aligned[i]
        breakout_short = close[i] < camarilla_l5_aligned[i]
        
        # Re-entry conditions (price back inside Camarilla H3-L3 range)
        # Calculate Camarilla H3/L3 for the day: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
        camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
        camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        price_in_range = (close[i] > camarilla_l3_aligned[i]) and (close[i] < camarilla_h3_aligned[i])
        
        if position == 0:
            # Long: breakout above H5 AND close > 1d EMA50 AND CMF > 0
            if breakout_long and close[i] > ema50_1d_aligned[i] and cmf_20_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: breakout below L5 AND close < 1d EMA50 AND CMF < 0
            elif breakout_short and close[i] < ema50_1d_aligned[i] and cmf_20_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: breakout below L4 OR price re-enters Camarilla H3-L3 range
            if breakout_short or price_in_range:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout above H4 OR price re-enters Camarilla H3-L3 range
            if breakout_long or price_in_range:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H4_H5_Breakout_1dTrend_ChaikinVolume_v1"
timeframe = "12h"
leverage = 1.0