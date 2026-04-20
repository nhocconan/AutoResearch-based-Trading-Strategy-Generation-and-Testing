#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_Trend_v1
Concept: 4h Donchian(20) breakout with volume spike and EMA50 trend filter.
- Long: Close > DonchianHigh(20) AND volume > 1.5 * SMA(volume,20) AND EMA50 rising
- Short: Close < DonchianLow(20) AND volume > 1.5 * SMA(volume,20) AND EMA50 falling
- Exit: Opposite Donchian breakout (long exits on DonchianLow, short exits on DonchianHigh)
- Position sizing: 0.25
- Target: 20-50 trades/year (80-200 total over 4 years)
Works in bull/bear: Donchian captures breakouts, volume confirms institutional interest, EMA50 filters counter-trend noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_VolumeSpike_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h: Donchian channels (20-period) ===
    # Donchian High: highest high of last 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian Low: lowest low of last 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h: EMA50 trend filter ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Volume spike filter (volume > 1.5 * 20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above Donchian High AND volume spike AND EMA50 rising
            if close[i] > donchian_high[i] and volume_spike[i] and ema50[i] > ema50[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Donchian Low AND volume spike AND EMA50 falling
            elif close[i] < donchian_low[i] and volume_spike[i] and ema50[i] < ema50[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below Donchian Low (opposite breakout)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above Donchian High (opposite breakout)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals