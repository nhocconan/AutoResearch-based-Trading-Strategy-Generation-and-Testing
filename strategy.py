#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 1d trend filter
# Williams Alligator (Jaw=TEETH=LIPS) identifies trend phases and avoids whipsaws
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# 1d EMA50 trend filter ensures alignment with higher timeframe direction
# Volume confirmation (1.5x 20-period average) filters low-quality signals
# Designed for 6h timeframe to capture medium-term swings in both bull and bear markets
# Target: 80-180 total trades over 4 years (20-45/year)

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams Alligator on 6h: Jaw=13, Teeth=8, Lips=5 (all SMMA)
    # SMMA calculation using EMA as approximation (standard practice)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values  # Jaw (Blue)
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values    # Teeth (Red)
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values    # Lips (Green)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and Elder Ray)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator trend detection:
        # Uptrend: Lips > Teeth > Jaw
        # Downtrend: Jaw > Teeth > Lips
        # Avoid trading when Alligator is sleeping (intertwined)
        alligator_uptrend = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_downtrend = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator uptrend + Bull Power > 0 + price > 1d EMA50 + volume spike
            if (alligator_uptrend and bull_power[i] > 0 and 
                close[i] > ema_50_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + Bear Power > 0 + price < 1d EMA50 + volume spike
            elif (alligator_downtrend and bear_power[i] > 0 and 
                  close[i] < ema_50_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns down OR Bear Power becomes negative
            if not alligator_uptrend or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns up OR Bull Power becomes negative
            if not alligator_downtrend or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals