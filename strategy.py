#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (H3, L3) from 1w for breakout signals (wider bands = fewer false breaks)
- 1w EMA34 as trend filter (long only above, short only below) - slower MA avoids whipsaw
- Volume > 1.8x 24-period average for confirmation (adjusts for 12h lower frequency)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts
- Uses 1w HTF as specified in experiment parameters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.8x 24-period average (adjusted for 12h lower frequency)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1w data for Camarilla pivot calculation (HTF as specified)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels (H3, L3) from prior 1w bar
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H3 = close + (range * 1.1 / 4)
    # L3 = close - (range * 1.1 / 4)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    rng = high_1w - low_1w
    camarilla_h3 = close_1w + (rng * 1.1 / 4.0)
    camarilla_l3 = close_1w - (rng * 1.1 / 4.0)
    
    # Align Camarilla levels to 1w timeframe (using completed 1w bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # 1w data for EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 34)  # Volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_aligned[i]  # Close above H3
        breakout_down = close[i] < camarilla_l3_aligned[i]  # Close below L3
        
        if position == 0:
            # Long: 1w Camarilla H3 breakout up AND price above 1w EMA34 AND volume confirmation
            if breakout_up and close[i] > ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 1w Camarilla L3 breakout down AND price below 1w EMA34 AND volume confirmation
            elif breakout_down and close[i] < ema_34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1w Camarilla L3 breakdown OR price crosses below 1w EMA34
            if breakout_down or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1w Camarilla H3 breakout OR price crosses above 1w EMA34
            if breakout_up or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA34_VolumeSpike_Filter_v1"
timeframe = "12h"
leverage = 1.0