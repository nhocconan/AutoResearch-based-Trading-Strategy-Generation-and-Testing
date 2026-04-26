#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter_v1
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) combined with 1d EMA50 trend filter. Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) with 1d uptrend. Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) with 1d downtrend. Uses discrete position sizing (0.25) to minimize fee drag and allow for reversal signals. Designed for low trade frequency (<30/year) to work in both bull and bear markets by capturing momentum shifts with trend confirmation.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power > 0 = bullish momentum
    bear_power = ema_13 - low   # Bear Power > 0 = bearish momentum
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1d EMA50 (50), 6h EMA13 (13)
    start_idx = max(50, 13)
    
    for i in range(start_idx, n):
        # Skip if 1d trend filter not ready
        if np.isnan(ema_50_1d_aligned[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum) AND Bear Power < 0 (not bearish) AND 1d uptrend (close > EMA50)
            long_signal = (bull_val > 0) and (bear_val < 0) and (close_val > ema_50_1d_val)
            # Short: Bear Power > 0 (bearish momentum) AND Bull Power < 0 (not bullish) AND 1d downtrend (close < EMA50)
            short_signal = (bear_val > 0) and (bull_val < 0) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: momentum shifts bearish (Bear Power > 0 AND Bull Power < 0) OR trend reversal (close < EMA50)
            if (bear_val > 0 and bull_val < 0) or (close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: momentum shifts bullish (Bull Power > 0 AND Bear Power < 0) OR trend reversal (close > EMA50)
            if (bull_val > 0 and bear_val < 0) or (close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0