#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend direction.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h).
- Trend filter: 12h EMA50 slope (rising/falling) determines bias - only take longs in uptrend, shorts in downtrend.
- Entry: Long when Bull Power > 0 AND rising AND volume > 1.5 * 20-period volume MA.
         Short when Bear Power < 0 AND falling AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit).
- Volume confirmation: avoids weak breakouts.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull (trend-following longs) and bear (trend-following shorts) via 12h EMA50 filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 13-period EMA for Elder Ray (on 6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA50 slope (trend direction) - rising if current > previous
    ema50_slope = np.zeros_like(ema50_12h)
    ema50_slope[1:] = ema50_12h[1:] > ema50_12h[:-1]  # True if rising
    
    # Align 12h EMA50 and slope to 6h
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope.astype(float))
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 50, 20)  # Need enough bars for EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        is_uptrend = ema50_slope_aligned[i] > 0.5  # Boolean: True if rising
        is_downtrend = ema50_slope_aligned[i] < 0.5  # Boolean: True if falling
        vol_ok = volume_spike[i]
        
        if position == 0:
            # Check for entry signals
            if vol_ok:
                # Long: Bull Power > 0 AND uptrend
                if curr_bull > 0 and is_uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 AND downtrend
                elif curr_bear < 0 and is_downtrend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power < 0 (momentum fading)
            if curr_bull < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power > 0 (momentum fading)
            if curr_bear > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0