#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Long when: Price breaks above 20-period Donchian high AND 1d ATR(14)/ATR(50) < 0.8 (low volatility regime) AND 1d volume > 1.3x 20-period average
# Short when: Price breaks below 20-period Donchian low AND 1d ATR(14)/ATR(50) < 0.8 AND 1d volume > 1.3x 20-period average
# Exit when price touches opposite Donchian level (e.g., long exits at Donchian low)
# Donchian provides clear structure, low volatility regime reduces whipsaw, volume confirms institutional interest
# Target: 80-160 total trades over 4 years (20-40/year) with discrete sizing 0.25

name = "4h_Donchian20_ATRRegime_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR regime and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: ATR(14)/ATR(50) < 0.8 indicates low volatility (regime filter)
    atr_ratio = np.where(atr_50 != 0, atr_14 / atr_50, 1.0)
    atr_ratio = np.where(np.isnan(atr_ratio), 1.0, atr_ratio)  # Default to high vol if NaN
    low_vol_regime = atr_ratio < 0.8
    
    # Calculate 1d volume spike (current volume > 1.3x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * vol_ma_20)
    
    # Align HTF indicators to 4h
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Calculate 4h Donchian channels (20-period)
    donch_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_h[i]) or np.isnan(donch_l[i]) or 
            np.isnan(low_vol_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike_aligned[i])
        low_vol_cond = bool(low_vol_aligned[i])
        
        if position == 0:
            # Long: Break above Donchian high in low vol regime with volume spike
            if close[i] > donch_h[i] and low_vol_cond and vol_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low in low vol regime with volume spike
            elif close[i] < donch_l[i] and low_vol_cond and vol_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch Donchian low
            if close[i] <= donch_l[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch Donchian high
            if close[i] >= donch_h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals