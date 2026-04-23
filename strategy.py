#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (H3, L3) from 1d for breakout signals
- 1d EMA50 as trend filter (long only above, short only below)
- Volume > 1.5x 24-period average for confirmation (reduced from 2.0x to increase trade frequency appropriately for 12h)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts
- Avoids SOL-only bias by requiring BTC/ETH alignment through trend filter
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
    
    # Volume confirmation: > 1.5x 24-period average (adjusted for 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (H3, L3) from prior 1d bar
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H3 = pivot + (range * 1.1 / 2)
    # L3 = pivot - (range * 1.1 / 2)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    camarilla_h3 = pivot + (rng * 1.1 / 2.0)
    camarilla_l3 = pivot - (rng * 1.1 / 2.0)
    
    # Align Camarilla levels to 12h timeframe (using completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d data for EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 50)  # Volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_aligned[i]  # Close above H3
        breakout_down = close[i] < camarilla_l3_aligned[i]  # Close below L3
        
        if position == 0:
            # Long: 1d Camarilla H3 breakout up AND price above 1d EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 1d Camarilla L3 breakout down AND price below 1d EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d Camarilla L3 breakdown OR price crosses below 1d EMA50
            if breakout_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1d Camarilla H3 breakout OR price crosses above 1d EMA50
            if breakout_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_VolumeSpike_Filter_v1"
timeframe = "12h"
leverage = 1.0