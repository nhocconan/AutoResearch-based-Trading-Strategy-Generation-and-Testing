#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_v2
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with EMA65 regime filter.
Long when Bull Power > 0 AND Bear Power < 0 AND price > EMA65 (bullish regime + bullish energy).
Short when Bull Power < 0 AND Bear Power > 0 AND price < EMA65 (bearish regime + bearish energy).
Volume confirmation to avoid low-activity false signals.
Designed to work in bull markets (catches strong uptrends) and bear markets (catches strong downtrends) by requiring alignment with EMA65 regime.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 and EMA65 on 6h
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema65 = close_s.ewm(span=65, adjust=False, min_periods=65).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA65(65), EMA13(13), volume MA(20)
    start_idx = max(65, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema13[i]) or np.isnan(ema65[i]) or np.isnan(vol_ma[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND price > EMA65 AND volume confirm
            long_signal = (bull_power[i] > 0) and (bear_power[i] < 0) and (close[i] > ema65[i]) and vol_conf
            
            # Short: Bull Power < 0 AND Bear Power > 0 AND price < EMA65 AND volume confirm
            short_signal = (bull_power[i] < 0) and (bear_power[i] > 0) and (close[i] < ema65[i]) and vol_conf
            
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
            # Exit: regime fails (price < EMA65) OR energy fades (Bull Power <= 0 OR Bear Power >= 0)
            if (close[i] < ema65[i]) or (bull_power[i] <= 0) or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: regime fails (price > EMA65) OR energy fades (Bull Power >= 0 OR Bear Power <= 0)
            if (close[i] > ema65[i]) or (bull_power[i] >= 0) or (bear_power[i] <= 0):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_v2"
timeframe = "6h"
leverage = 1.0