#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Alligator regime filter with volume confirmation.
# Elder Ray measures bull/bear power via EMA13. Alligator (Jaw/Teeth/Lips) defines regime: 
#   - Trending when Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear)
#   - Ranging when lines intertwine
# Only take Elder Ray signals aligned with Alligator regime. Volume confirms participation.
# Works in bull (buy bull power with bullish Alligator) and bear (sell bear power with bearish Alligator).
# Discrete sizing 0.25 targets 50-150 trades over 4 years.

name = "6h_ElderRay_Alligator_Regime_1d_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Alligator (SMAs: Jaw=13, Teeth=8, Lips=5)
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d ATR(14) for Elder Ray EMA smoothing
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 6h EMA13 for Elder Ray (using ATR as proxy for price series)
    # Elder Ray: Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
    # We'll use ATR-based EMA as a volatility-adjusted proxy
    ema13_atr = pd.Series(atr_14_1d_aligned).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13_atr
    bear_power = ema13_atr - low
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 6h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(20, 13) + 1  # 21 (for volume MA20 and EMA13)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(ema13_atr[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        # Alligator regime: trending when lips, teeth, jaw are aligned
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray signals
        bull_power_signal = bull_power[i] > 0  # Bullish momentum
        bear_power_signal = bear_power[i] > 0  # Bearish momentum
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull power AND bullish Alligator AND volume confirmation
            if bull_power_signal and bullish_alligator and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bear power AND bearish Alligator AND volume confirmation
            elif bear_power_signal and bearish_alligator and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish Alligator (regime change) or loss of bull power
            if not bullish_alligator or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish Alligator (regime change) or loss of bear power
            if not bearish_alligator or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals