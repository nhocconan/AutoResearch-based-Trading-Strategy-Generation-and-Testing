#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Uses Donchian channel (20-period high/low) from 1d for breakout signals
- 1w EMA50 as trend filter (long only above, short only below) - slower MA avoids whipsaw
- Volume > 2.0x 20-period average for confirmation (adjusts for 1d lower frequency)
- Position size: 0.25 discrete level to minimize fee churn
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
    
    # 1d data for Donchian channel calculation (primary timeframe)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
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
    
    # 1w data for EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 20, 50)  # Donchian, Volume MA, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_high[i]  # Close above upper Donchian
        breakout_down = close[i] < lowest_low[i]  # Close below lower Donchian
        
        if position == 0:
            # Long: Donchian breakout up AND price above 1w EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND price below 1w EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakdown OR price crosses below 1w EMA50
            if breakout_down or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout OR price crosses above 1w EMA50
            if breakout_up or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_VolumeSpike_Filter_v1"
timeframe = "1d"
leverage = 1.0