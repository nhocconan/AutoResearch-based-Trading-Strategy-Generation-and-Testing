#!/usr/bin/env python3
"""
12h_Camarilla_H4_Breakout_1wTrend_VolumeSpike
Hypothesis: 12h Camarilla H4/L4 breakout with 1w trend filter (price > 1w EMA34 for long, < for short) and volume spike confirmation.
Uses tighter H4/L4 levels vs R3/S3 to reduce false breakouts and lower trade frequency. 1w EMA34 ensures trend alignment.
Volume spike > 1.5x 20-period average confirms institutional participation. Exits on opposite H4/L4 break or re-entry into H3/L3 range.
Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years) with controlled risk.
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
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.1*(high-low), L4 = close - 1.1*(high-low)
    # H3 = close + 1.0*(high-low), L3 = close - 1.0*(high-low)
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.0 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
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
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
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
        breakout_long = close[i] > camarilla_h4_aligned[i]
        breakout_short = close[i] < camarilla_l4_aligned[i]
        
        # Re-entry conditions (price back inside Camarilla H3-L3 range)
        price_in_range = (close[i] > camarilla_l3_aligned[i]) and (close[i] < camarilla_h3_aligned[i])
        
        if position == 0:
            # Long: breakout above H4 AND close > 1w EMA34 AND volume spike
            if breakout_long and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below L4 AND close < 1w EMA34 AND volume spike
            elif breakout_short and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
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

name = "12h_Camarilla_H4_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0