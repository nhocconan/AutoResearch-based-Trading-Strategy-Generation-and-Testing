#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend when lines are aligned and separated.
# In bullish trend: Lips > Teeth > Jaw; in bearish: Lips < Teeth < Jaw.
# Combined with 1d EMA50 for higher-timeframe trend alignment and volume spike for confirmation.
# Works in both bull and bear markets by only taking trades in direction of higher-timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.

name = "6h_WilliamsAlligator_1dEMA50_Volume"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # All values shifted forward by their respective offsets
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Calculate SMAs
    sma_jaw = pd.Series(close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    sma_teeth = pd.Series(close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    sma_lips = pd.Series(close).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Apply shifts (align to close price)
    jaw = np.roll(sma_jaw, jaw_shift)
    teeth = np.roll(sma_teeth, teeth_shift)
    lips = np.roll(sma_lips, lips_shift)
    
    # Invalid values due to roll and insufficient data
    jaw[:jaw_shift] = np.nan
    teeth[:teeth_shift] = np.nan
    lips[:lips_shift] = np.nan
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and 1d EMA)
    start_idx = max(jaw_shift + jaw_period, teeth_shift + teeth_period, lips_shift + lips_period, 50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator: Lips > Teeth > Jaw (alligator awake, eating up)
            bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            # Bearish Alligator: Lips < Teeth < Jaw (alligator awake, eating down)
            bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long entry: bullish Alligator AND price > 1d EMA50 AND volume spike
            if bullish_alligator and (close[i] > ema_50_1d_aligned[i]) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish Alligator AND price < 1d EMA50 AND volume spike
            elif bearish_alligator and (close[i] < ema_50_1d_aligned[i]) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR price < 1d EMA50 (trend change)
            bearish_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            if bearish_alligator or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR price > 1d EMA50 (trend change)
            bullish_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            if bullish_alligator or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals