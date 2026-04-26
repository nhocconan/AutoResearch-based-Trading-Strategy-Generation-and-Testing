#!/usr/bin/env python3
"""
1d_Camarilla_H4_H5_Breakout_1wTrendFilter_v1
Hypothesis: Daily Camarilla H4/H5 breakout with 1-week EMA50 trend filter and volume spike confirmation.
- Uses 1d timeframe for low trade frequency (target: 30-100 total trades over 4 years)
- Camarilla H4/H5 levels from prior day provide strong support/resistance
- 1-week EMA50 ensures trades align with higher timeframe trend (works in bull/bear)
- Volume spike (1.8x 20-period average) confirms institutional participation
- Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the 1w trend and using volume to filter false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop (for Camarilla levels)
    df_1d = get_htf_data(prices, '1d')
    
    # Load 1w data ONCE before loop (for EMA50 trend filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    camarilla_h4 = df_1d['close'].values + (1.0/6) * (df_1d['high'].values - df_1d['low'].values)
    camarilla_h5 = df_1d['close'].values + (1.0/4) * (df_1d['high'].values - df_1d['low'].values)
    camarilla_l4 = df_1d['close'].values - (1.0/6) * (df_1d['high'].values - df_1d['low'].values)
    camarilla_l5 = df_1d['close'].values - (1.0/4) * (df_1d['high'].values - df_1d['low'].values)
    
    # Align Camarilla levels to 1d timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Volume spike: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h5_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or
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
        breakout_long = close[i] > camarilla_h5_aligned[i]
        breakout_short = close[i] < camarilla_l5_aligned[i]
        
        # Re-entry conditions (price back inside Camarilla H3-L3 range)
        # Calculate Camarilla H3/L3 for the day: H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
        camarilla_h3 = df_1d['close'].values + 1.1 * (df_1d['high'].values - df_1d['low'].values)
        camarilla_l3 = df_1d['close'].values - 1.1 * (df_1d['high'].values - df_1d['low'].values)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        price_in_range = (close[i] > camarilla_l3_aligned[i]) and (close[i] < camarilla_h3_aligned[i])
        
        if position == 0:
            # Long: breakout above H5 AND close > 1w EMA50 AND volume spike
            if breakout_long and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below L5 AND close < 1w EMA50 AND volume spike
            elif breakout_short and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
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

name = "1d_Camarilla_H4_H5_Breakout_1wTrendFilter_v1"
timeframe = "1d"
leverage = 1.0