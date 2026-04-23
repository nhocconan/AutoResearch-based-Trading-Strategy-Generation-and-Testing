#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power + Weekly Trend Filter + Volume Spike
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
- Long: Bull Power > 0 + Bear Power < 0 (bullish momentum) + Weekly EMA34 uptrend + Volume > 2.0x 28-period avg
- Short: Bull Power < 0 + Bear Power > 0 (bearish momentum) + Weekly EMA34 downtrend + Volume > 2.0x 28-period avg
- Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power < 0 for short exit)
- Weekly EMA34 provides strong trend alignment to reduce whipsaws
- Volume confirmation ensures breakout validity
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Works in bull (trend continuation) and bear (mean reversion via momentum divergence)
- Discrete position sizing: ±0.25 to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 28-period average (tight spike filter)
    vol_ma = pd.Series(volume).rolling(window=28, min_periods=28).mean().values
    
    # Calculate Elder Ray components on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(28, 13, 34)  # Need 28 for volume MA, 13 for EMA13, 34 for weekly EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        # Elder Ray signals
        bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0
        bearish_momentum = bull_power[i] < 0 and bear_power[i] > 0
        
        if position == 0:
            # Long: bullish momentum + weekly uptrend + volume spike
            if volume_spike and bullish_momentum and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + weekly downtrend + volume spike
            elif volume_spike and bearish_momentum and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish momentum (momentum divergence)
            if bear_power[i] > 0:  # Bear power turned positive
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish momentum (momentum divergence)
            if bull_power[i] > 0:  # Bull power turned positive
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0