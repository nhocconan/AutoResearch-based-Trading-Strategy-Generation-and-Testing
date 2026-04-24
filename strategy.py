#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume spike confirmation.
- Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend direction via alignment.
- 1d EMA50 ensures we trade only in the direction of the daily trend, reducing whipsaws.
- Volume spike (>2.0x 20-bar average) confirms institutional participation.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 50-150 total over 4 years (12-37/year) to minimize fee drag.
- Works in bull/bear markets via daily trend filter and Alligator's trend-following nature.
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
    
    # Get 1d data ONCE before loop for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs with specific periods and shifts
    # JAW: 13-period SMMA shifted 8 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # TEETH: 8-period SMMA shifted 5 bars forward
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # LIPS: 5-period SMMA shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13, 8, 5, 20) + 8  # Need enough for EMA and Alligator (max shift 8)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only trade if volume confirms
            if volume_confirm:
                # Alligator long: Lips > Teeth > Jaw (bullish alignment) AND price above 1d EMA50
                if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Alligator short: Lips < Teeth < Jaw (bearish alignment) AND price below 1d EMA50
                elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR price crosses below 1d EMA50
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR price crosses above 1d EMA50
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0