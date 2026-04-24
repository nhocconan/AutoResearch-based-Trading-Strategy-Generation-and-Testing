#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 6h for Elder Ray calculation and entries/exits.
- HTF: 12h EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h).
- Volume: Current 6h volume > 1.5 * 20-period volume MA to avoid low-volatility false signals.
- Entry Long: Bull Power > 0 AND Bear Power < 0 (bullish market structure) AND 12h EMA trend bullish AND volume spike.
- Entry Short: Bull Power < 0 AND Bear Power > 0 (bearish market structure) AND 12h EMA trend bearish AND volume spike.
- Exit: Opposite Elder Ray condition or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray shows the power of bulls and bears; combined with trend filter and volume, it avoids whipsaws.
Works in both bull and bear markets by requiring alignment of microstructure, trend, and participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    df_12h_close = df_12h['close'].values
    ema34_12h = pd.Series(df_12h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 12h for confirmation
    df_12h_volume = df_12h['volume'].values
    vol_ma_12h = pd.Series(df_12h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period 12h volume MA
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)  # Need enough bars for EMA13, EMA34, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema34_val = ema34_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Bull Power > 0 AND Bear Power < 0 (bulls in control, bears weak) AND 12h EMA bullish
                if bull_power[i] > 0 and bear_power[i] < 0 and curr_close > ema34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Bull Power < 0 AND Bear Power > 0 (bears in control, bulls weak) AND 12h EMA bearish
                elif bull_power[i] < 0 and bear_power[i] > 0 and curr_close < ema34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Elder Ray turns bearish OR loss of volume confirmation
            if (bull_power[i] <= 0 or bear_power[i] >= 0) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Elder Ray turns bullish OR loss of volume confirmation
            if (bull_power[i] >= 0 or bear_power[i] <= 0) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0