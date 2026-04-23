#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Uses Camarilla pivot levels (H4, L4) from 1w for breakout signals (wider bands = fewer false breaks)
- 1w EMA50 as trend filter (long only above, short only below) - slower MA avoids whipsaw
- Volume > 2.0x 20-period average for confirmation (adjusts for 1d lower frequency)
- Position size: 0.30 discrete level to minimize fee churn
- Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
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
    
    # Volume confirmation: > 2.0x 20-period average (adjusted for 1d lower frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w data for Camarilla pivot calculation (HTF as specified)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels (H4, L4) from prior 1w bar
    # Pivot = (high + low + close) / 3
    # Range = high - low
    # H4 = close + (range * 1.1 / 2)
    # L4 = close - (range * 1.1 / 2)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    rng = high_1w - low_1w
    camarilla_h4 = close_1w + (rng * 1.1 / 2.0)
    camarilla_l4 = close_1w - (rng * 1.1 / 2.0)
    
    # Align Camarilla levels to 1w timeframe (using completed 1w bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # 1w data for EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h4_aligned[i]  # Close above H4
        breakout_down = close[i] < camarilla_l4_aligned[i]  # Close below L4
        
        if position == 0:
            # Long: 1w Camarilla H4 breakout up AND price above 1w EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: 1w Camarilla L4 breakout down AND price below 1w EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: 1w Camarilla L4 breakdown OR price crosses below 1w EMA50
            if breakout_down or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: 1w Camarilla H4 breakout OR price crosses above 1w EMA50
            if breakout_up or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_Camarilla_H4L4_Breakout_1wEMA50_VolumeSpike_Filter_v1"
timeframe = "1d"
leverage = 1.0