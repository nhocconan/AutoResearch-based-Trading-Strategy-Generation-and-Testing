#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: 1d Camarilla R1/S1 breakout with 1-week EMA50 trend filter and volume spike confirmation.
Enters long when price breaks above R1 (close + 1.1*(high-low)) AND close > 1w EMA50 AND volume > 2.0 * 20-period average volume.
Enters short when price breaks below S1 (close - 1.1*(high-low)) AND close < 1w EMA50 AND volume > 2.0 * 20-period average volume.
Exits on opposite Camarilla level touch (S1 for long, R1 for short) or when price re-enters the Camarilla H3-L3 range.
Uses 1w EMA50 for higher timeframe trend alignment to avoid counter-trend trades.
Volume spike confirms institutional participation. Camarilla levels provide mathematically derived support/resistance.
Designed for 1d timeframe to target 7-25 trades/year (30-100 total over 4 years).
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
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: R1 = close + 1.1*(high-low), S1 = close - 1.1*(high-low)
    # We use R1/S1 as breakout levels (more conservative than H4/L4)
    camarilla_r1 = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_s1 = close_1w - 1.1 * (high_1w - low_1w)
    
    # Align Camarilla levels to 1d timeframe (use previous week's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
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
        # Calculate Camarilla H3-L3 for the week: H3 = close + 1.2*(high-low), L3 = close - 1.2*(high-low)
        camarilla_h3 = close_1w + 1.2 * (high_1w - low_1w)
        camarilla_l3 = close_1w - 1.2 * (high_1w - low_1w)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
        price_in_range = (close[i] > camarilla_l3_aligned[i]) and (close[i] < camarilla_h3_aligned[i])
        
        if position == 0:
            # Long: breakout above R1 AND close > 1w EMA50 AND volume spike
            if breakout_long and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 AND close < 1w EMA50 AND volume spike
            elif breakout_short and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
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

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0