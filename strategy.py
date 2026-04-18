#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ATR filter + Volume spike
# Donchian breakouts capture strong momentum moves with clear risk management.
# 1d ATR filter ensures we only trade when volatility is sufficient (avoid chop).
# Volume spike confirms institutional participation in the breakout.
# Works in both bull and bear markets by trading breakouts in direction of trend.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "4h_Donchian20_1dATRFilter_VolumeSpike"
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
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) with Wilder's smoothing
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[:14])  # Simple average for first 14
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    atr_14_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_filter = atr_14 > (0.5 * atr_14_ma_20)  # Only trade when volatility is above half of 20-period average
    
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_filter_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_spike = volume_spike[i]
        atr_ok = atr_filter_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian band with volume spike and ATR filter
            if close_val > upper and vol_spike and atr_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band with volume spike and ATR filter
            elif close_val < lower and vol_spike and atr_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price closes below lower Donchian band (reversal)
            if close_val < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price closes above upper Donchian band (reversal)
            if close_val > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals