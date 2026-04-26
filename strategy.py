#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
Enters long when price breaks above R1 (close + 1.1*(high-low)) AND close > 1d EMA34 AND volume > 1.5 * 20-period average volume.
Enters short when price breaks below S1 (close - 1.1*(high-low)) AND close < 1d EMA34 AND volume > 1.5 * 20-period average volume.
Exits on opposite Camarilla level touch (S1 for long, R1 for short) or when price re-enters the Camarilla H3-L3 range.
Uses 1d EMA34 for higher timeframe trend alignment to avoid counter-trend trades.
Volume spike confirms institutional participation. Camarilla levels provide mathematically derived support/resistance.
Designed for 4h timeframe to target 20-50 trades/year (75-200 total over 4 years).
Works in bull/bear markets by trading with the 1d trend and using volume to filter false breakouts.
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
    
    # Camarilla levels: R1 = close + 1.1*(high-low), S1 = close - 1.1*(high-low)
    # R3/S3 are too wide; R1/S1 provide tighter, more frequent breakouts
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)  # Same as R1
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)  # Same as S1
    # Actually, H3/L3 are R1/S1, H4/L4 are R2/S2, H5/L5 are R3/S3
    # For re-entry, we use H3/L3 (R1/S1) as the range boundaries
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_h3_aligned = camarilla_r1_aligned  # H3 = R1
    camarilla_l3_aligned = camarilla_s1_aligned  # L3 = S1
    
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
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
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
        breakout_long = close[i] > camarilla_r1_aligned[i]
        breakout_short = close[i] < camarilla_s1_aligned[i]
        
        # Re-entry conditions (price back inside Camarilla H3-L3 range)
        price_in_range = (close[i] > camarilla_l3_aligned[i]) and (close[i] < camarilla_h3_aligned[i])
        
        if position == 0:
            # Long: breakout above R1 AND close > 1d EMA34 AND volume spike
            if breakout_long and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 AND close < 1d EMA34 AND volume spike
            elif breakout_short and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: breakout below S1 OR price re-enters Camarilla H3-L3 range
            if breakout_short or price_in_range:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout above R1 OR price re-enters Camarilla H3-L3 range
            if breakout_long or price_in_range:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0