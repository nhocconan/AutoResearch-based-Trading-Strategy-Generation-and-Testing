#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d regime filter
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low (1d timeframe)
# - Bull regime: Bull Power > 0 and rising (current > previous)
# - Bear regime: Bear Power > 0 and rising (current > previous)
# - Entry: 6h close crosses above/below 6-period EMA in direction of 1d regime
# - Exit: Opposite crossover or regime change
# - Volume filter: 6h volume > 1.5x 20-period average for confirmation
# - Works in bull/bear by following 1d Elder Ray regime
# - Target: 20-40 trades/year to avoid excessive fee drag

name = "6h_ElderRay_1dRegime_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 on 1d close
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = ema13_1d - df_1d['low'].values
    
    # Regime: Bull if Bull Power > 0 and rising, Bear if Bear Power > 0 and rising
    bull_regime_1d = (bull_power_1d > 0) & (np.roll(bull_power_1d, 1) < bull_power_1d)
    bear_regime_1d = (bear_power_1d > 0) & (np.roll(bear_power_1d, 1) < bear_power_1d)
    
    # Align regimes to 6h
    bull_regime_aligned = align_htf_to_ltf(prices, df_1d, bull_regime_1d.astype(float))
    bear_regime_aligned = align_htf_to_ltf(prices, df_1d, bear_regime_1d.astype(float))
    
    # 6h EMA6 for entry timing
    ema6_6h = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # 6h volume average (20-period)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_regime_aligned[i]) or np.isnan(bear_regime_aligned[i]) or 
            np.isnan(ema6_6h[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x average
        volume_filter = vol_ma_6h[i] > 0 and volume[i] > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Look for long entry: bull regime + price crosses above EMA6 + volume
            if bull_regime_aligned[i] > 0.5 and close[i] > ema6_6h[i] and close[i-1] <= ema6_6h[i-1] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: bear regime + price crosses below EMA6 + volume
            elif bear_regime_aligned[i] > 0.5 and close[i] < ema6_6h[i] and close[i-1] >= ema6_6h[i-1] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on bear regime or price crosses below EMA6
            if bear_regime_aligned[i] > 0.5 or (close[i] < ema6_6h[i] and close[i-1] >= ema6_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on bull regime or price crosses above EMA6
            if bull_regime_aligned[i] > 0.5 or (close[i] > ema6_6h[i] and close[i-1] <= ema6_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals