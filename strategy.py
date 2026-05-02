#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams Alligator with 1d EMA50 trend filter
# Uses 6h primary timeframe for lower trade frequency (target: 50-150 trades over 4 years)
# Elder Ray (Bull/Bear Power) measures trend strength via EMA13 deviation
# Williams Alligator (Jaw/Teeth/Lips) provides trend direction and avoids whipsaws
# 1d EMA50 filter ensures alignment with higher timeframe momentum
# Designed for both bull and bear markets: long when Bull Power > 0 and price > Alligator Lips in uptrend,
# short when Bear Power < 0 and price < Alligator Lips in downtrend
# Volume confirmation (1.5x 20-period average) filters low-participation moves
# Discrete sizing: 0.25 for positions, 0.0 for flat
# Target: 75-125 total trades over 4 years (19-31/year) - within proven winning range for 6h

name = "6h_ElderRay_Alligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)
    alligator_jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    alligator_teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    alligator_lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator Lips and HTF data alignment)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(alligator_lips[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Uptrend condition: price > Alligator Lips and EMA50 rising
            uptrend = close[i] > alligator_lips[i] and ema_50_aligned[i] > ema_50_aligned[i-1]
            # Downtrend condition: price < Alligator Lips and EMA50 falling
            downtrend = close[i] < alligator_lips[i] and ema_50_aligned[i] < ema_50_aligned[i-1]
            
            # Long: Bull Power > 0 + uptrend + volume confirmation
            if bull_power[i] > 0 and uptrend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 + downtrend + volume confirmation
            elif bear_power[i] < 0 and downtrend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear Power < 0 or price breaks below Alligator Teeth
            if bear_power[i] < 0 or close[i] < alligator_teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power > 0 or price breaks above Alligator Teeth
            if bull_power[i] > 0 or close[i] > alligator_teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals