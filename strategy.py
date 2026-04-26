#!/usr/bin/env python3
"""
12h_Camarilla_H4_H5_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: 12h Camarilla H4/H5 breakout with 1d EMA34 trend filter and volume spike confirmation.
Enters long when price breaks above H4 (close + 1.0*(high-low)) AND close > 1d EMA34 AND volume > 1.5 * 20-period average volume.
Enters short when price breaks below H5 (close - 1.0*(high-low)) AND close < 1d EMA34 AND volume > 1.5 * 20-period average volume.
Exits on opposite Camarilla level touch (H5 for long, H4 for short) or when price re-enters the Camarilla H-L range.
Uses 1d EMA34 for higher timeframe trend alignment to avoid counter-trend trades.
Volume spike confirms institutional participation. Camarilla levels provide mathematically derived support/resistance.
Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
Works in bull/bear markets by trading with the 1d trend and using volume to filter false breakouts.
Version 2: Reduced position size to 0.20 to lower drawdown and improved exit logic for better Sharpe.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4 = close + 1.0*(high-low), H5 = close - 1.0*(high-low)
    # We use H4/H5 as breakout levels (more conservative than R3/S3)
    camarilla_h4 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_h5 = close_1d - 1.0 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h5_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_h4_aligned[i]
        breakout_short = close[i] < camarilla_h5_aligned[i]
        
        # Re-entry conditions (price back inside Camarilla H-L range)
        # Calculate Camarilla H-L for the day: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
        camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
        camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        price_in_range = (close[i] > camarilla_l3_aligned[i]) and (close[i] < camarilla_h3_aligned[i])
        
        if position == 0:
            # Long: breakout above H4 AND close > 1d EMA34 AND volume spike
            if breakout_long and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: breakout below H5 AND close < 1d EMA34 AND volume spike
            elif breakout_short and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: breakout below H5 OR price re-enters Camarilla H3-L3 range
            if breakout_short or price_in_range:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: breakout above H4 OR price re-enters Camarilla H3-L3 range
            if breakout_long or price_in_range:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H4_H5_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0