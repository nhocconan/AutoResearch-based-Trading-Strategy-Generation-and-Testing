#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA13 trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA13, providing directional momentum.
# Combined with 1d EMA13 trend filter to ensure alignment with higher timeframe trend.
# Volume > 1.5x 20-period EMA confirms institutional participation.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_ElderRay_EMA13_1dEMA13_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA13 trend and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components on 1d data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume EMA has enough data
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum) + price above 1d EMA13 + volume spike
            if (bull_power_aligned[i] > 0 and price > ema_13_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish momentum) + price below 1d EMA13 + volume spike
            elif (bear_power_aligned[i] < 0 and price < ema_13_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power becomes negative (momentum shift) or price below EMA13
            if (bear_power_aligned[i] < 0) or (price < ema_13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power becomes positive (momentum shift) or price above EMA13
            if (bull_power_aligned[i] > 0) or (price > ema_13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals