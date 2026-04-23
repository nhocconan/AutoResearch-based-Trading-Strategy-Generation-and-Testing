#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume confirmation.
- Elder Ray Bull Power = High - EMA(close,34); Bear Power = EMA(close,34) - Low
- Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-period avg
- Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-period avg
- Exit: Opposite Elder Ray signal (Bull Power <= 0 for long exit, Bear Power <= 0 for short exit)
- Uses 1d EMA34 for trend alignment to avoid counter-trend trades, volume confirmation for strength
- Works in bull markets (captures strong uptrends) and bear markets (captures strong downtrends)
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to balance return and minimize fee churn
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Elder Ray components: need EMA34 of close
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    bull_power = high - ema34  # Bull Power = High - EMA34
    bear_power = ema34 - low   # Bear Power = EMA34 - Low
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema34[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND uptrend AND volume
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema34_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND downtrend AND volume
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (loss of bullish momentum)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 (loss of bearish momentum)
            if bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0