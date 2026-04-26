#!/usr/bin/env python3
"""
6h_Camarilla_H4_H5_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 6h Camarilla H4/H5 breakout with 1d trend filter and 1d volume spike confirmation.
- Uses 6h timeframe for moderate trade frequency (target: 50-150 total trades over 4 years)
- Camarilla H4/H5 levels from 1d provide strong support/resistance from prior day
- 1d EMA50 filter ensures trades align with higher timeframe trend
- 1d volume spike (>2x 20-period average) confirms institutional participation
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the 1d trend and using volume spike to filter false breakouts
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
    
    # Calculate 1d volume spike confirmation (>2x 20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_h4 = close_1d + (1.0/6) * (high_1d - low_1d)  # H4 = close + 1/6*(high-low)
    camarilla_h5 = close_1d + (1.0/4) * (high_1d - low_1d)  # H5 = close + 1/4*(high-low)
    camarilla_l4 = close_1d - (1.0/6) * (high_1d - low_1d)  # L4 = close - 1/6*(high-low)
    camarilla_l5 = close_1d - (1.0/4) * (high_1d - low_1d)  # L5 = close - 1/4*(high-low)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or
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
            # Long: breakout above H5 AND close > 1d EMA50 AND volume spike
            if breakout_long and close[i] > ema50_1d_aligned[i] and vol_spike_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: breakout below L5 AND close < 1d EMA50 AND volume spike
            elif breakout_short and close[i] < ema50_1d_aligned[i] and vol_spike_aligned[i] > 0.5:
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

name = "6h_Camarilla_H4_H5_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0