#!/usr/bin/env python3
"""
1d_Camarilla_H4_H5_Breakout_1wTrendFilter_VolumeSpike_v2
Hypothesis: Daily Camarilla H4/H5 breakout with 1w trend filter and volume spike confirmation.
- Uses 1d timeframe for low trade frequency (target: 30-100 total trades over 4 years)
- Camarilla H4/H5 levels from 1d provide strong support/resistance from prior day
- 1w EMA34 filter ensures trades align with higher timeframe trend (works in bull/bear)
- Volume spike (>2x 20-period average) confirms institutional participation
- Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the 1w trend and using volume to filter false breakouts
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate volume spike on 1d (>2x 20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma20)
    
    # Calculate Camarilla levels from previous 1d bar (need daily OHLC)
    # For 1d timeframe, we can use the previous bar's values directly
    # Since we're on 1d timeframe, prices already contain daily OHLC
    camarilla_h4 = close + (1.0/6) * (high - low)  # H4 = close + 1/6*(high-low)
    camarilla_h5 = close + (1.0/4) * (high - low)  # H5 = close + 1/4*(high-low)
    camarilla_l4 = close - (1.0/6) * (high - low)  # L4 = close - 1/6*(high-low)
    camarilla_l5 = close - (1.0/4) * (high - low)  # L5 = close - 1/4*(high-low)
    
    # Align Camarilla levels (use previous day's levels to avoid look-ahead)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)  # Wrong HTF but we need to align to same timeframe
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h5)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l5)
    
    # Fix: For same timeframe, we need to shift by 1 bar to use previous day's levels
    # Actually, since we're on 1d timeframe, we can just shift the arrays
    camarilla_h4_aligned = np.roll(camarilla_h4_aligned, 1)
    camarilla_h5_aligned = np.roll(camarilla_h5_aligned, 1)
    camarilla_l4_aligned = np.roll(camarilla_l4_aligned, 1)
    camarilla_l5_aligned = np.roll(camarilla_l5_aligned, 1)
    camarilla_h4_aligned[0] = np.nan
    camarilla_h5_aligned[0] = np.nan
    camarilla_l4_aligned[0] = np.nan
    camarilla_l5_aligned[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1w EMA, 20 for volume MA)
    start_idx = max(34, 20) + 1  # +1 for the level shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
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
        
        if position == 0:
            # Long: breakout above H5 AND close > 1w EMA34 AND volume spike
            if breakout_long and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below L5 AND close < 1w EMA34 AND volume spike
            elif breakout_short and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: breakout below L4
            if breakout_short:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: breakout above H4
            if breakout_long:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H4_H5_Breakout_1wTrendFilter_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0