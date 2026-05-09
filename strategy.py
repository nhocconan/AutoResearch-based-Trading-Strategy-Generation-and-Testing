#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA13 trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and Bear Power < 0 (bullish), with volume spike and above 1d EMA13
# Short when Bear Power > 0 and Bull Power < 0 (bearish), with volume spike and below 1d EMA13
# Uses 1d EMA13 for trend filter to avoid counter-trend trades
# Volume > 1.3x 20-period EMA for institutional participation
# Designed to work in both bull and bear markets by following daily trend
# Target: 50-150 total trades over 4 years (12-37/year)
name = "6h_ElderRay_1dEMA13_Volume"
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
    
    # 1d data for EMA13 trend and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components on 1d: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d
    bear_power = ema_13_1d - low_1d
    
    # Align 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike filter: volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.3 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
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
            # Long: Bull Power > 0, Bear Power < 0 (bullish), volume spike, price above EMA13
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                vol_spike[i] and price > ema_13_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0, Bull Power < 0 (bearish), volume spike, price below EMA13
            elif (bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and 
                  vol_spike[i] and price < ema_13_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power becomes positive (bearish shift) or price below EMA13
            if bear_power_aligned[i] > 0 or price < ema_13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power becomes positive (bullish shift) or price above EMA13
            if bull_power_aligned[i] > 0 or price > ema_13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals