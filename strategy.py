#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA50 trend direction.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h).
- Trend filter: 1w EMA50 slope (rising/falling) determines bias: only long in uptrend, short in downtrend.
- Entry: Long when Bull Power > 0 AND rising AND volume > 1.5 * 20-period volume MA.
         Short when Bear Power < 0 AND falling AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit).
- Volume confirmation: avoids low-conviction moves.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear: trend filter ensures we only trade with the 1w trend,
  Elder Ray captures momentum within that trend, volume filters noise.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 and its slope (trend direction)
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    # Slope: current EMA - previous EMA (positive = rising, negative = falling)
    ema50_slope = np.diff(ema50_1w, prepend=ema50_1w[0])
    
    # Align 1w EMA50 and slope to 6h
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema50_slope)
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 20)  # Need enough 1w bars for EMA50, 6h bars for EMA13 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1w
        is_uptrend = ema50_slope_aligned[i] > 0  # Rising EMA50 = uptrend
        is_downtrend = ema50_slope_aligned[i] < 0  # Falling EMA50 = downtrend
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Long conditions: uptrend + Bull Power > 0 (bullish momentum)
                if is_uptrend and bull_power[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: downtrend + Bear Power < 0 (bearish momentum)
                elif is_downtrend and bear_power[i] < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative (momentum fading) OR trend shifts to downtrend
            if bull_power[i] < 0 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive (momentum fading) OR trend shifts to uptrend
            if bear_power[i] > 0 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1wEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0