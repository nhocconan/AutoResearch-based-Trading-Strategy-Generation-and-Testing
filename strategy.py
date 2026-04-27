# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day ATR-based volatility filter and 4-hour Donchian breakout.
# Combines volatility regime detection (low ATR = range, high ATR = trend) with price breakouts.
# In low volatility regimes, uses mean reversion at Bollinger Bands; in high volatility, follows breakouts.
# Volume confirmation filters breakouts. Designed for low trade frequency (<50/year) to minimize fee drag.
# Works in bull markets (breakout continuations) and bear markets (volatility contraction expansions).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR for volatility regime
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr1 = np.insert(tr1, 0, high_1d[0] - low_1d[0])
    atr14 = np.zeros_like(close_1d)
    atr14[0] = tr1[0]
    for i in range(1, len(tr1)):
        atr14[i] = (atr14[i-1] * 13 + tr1[i]) / 14
    
    # Calculate ATR ratio (current ATR / 50-period average ATR) for regime detection
    atr_ma = pd.Series(atr14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr14 / atr_ma
    atr_ratio = np.where(atr_ma > 0, atr_ratio, 1.0)
    
    # Align ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 4-hour indicators
    # Bollinger Bands (20, 2) for mean reversion in low volatility
    close_4h = pd.Series(close)
    bb_mid = close_4h.rolling(window=20, min_periods=20).mean().values
    bb_std = close_4h.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Donchian Channel (20) for breakouts in high volatility
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(bb_mid[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if atr_ratio_aligned[i] < 0.8:  # Low volatility regime - mean reversion
            # Long at lower Bollinger Band with rejection
            if (close[i] <= bb_lower[i] * 1.01 and 
                close[i] > bb_lower[i] and 
                i > start_idx and 
                close[i-1] <= bb_lower[i-1]):
                signals[i] = 0.25
                position = 1
            # Short at upper Bollinger Band with rejection
            elif (close[i] >= bb_upper[i] * 0.99 and 
                  close[i] < bb_upper[i] and 
                  i > start_idx and 
                  close[i-1] >= bb_upper[i-1]):
                signals[i] = -0.25
                position = -1
            # Exit mean reversion positions at midpoint
            elif position == 1 and close[i] >= bb_mid[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] <= bb_mid[i]:
                signals[i] = 0.0
                position = 0
        
        else:  # High volatility regime - follow breakouts
            # Long breakout: price breaks above Donchian high with volume
            if (close[i] > donch_high[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below Donchian low with volume
            elif (close[i] < donch_low[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            # Exit breakout positions on opposite Donchian touch
            elif position == 1 and close[i] <= donch_low[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] >= donch_high[i]:
                signals[i] = 0.0
                position = 0
        
        # Hold position
        if position == 1 and signals[i] == 0.0:
            signals[i] = 0.25
        elif position == -1 and signals[i] == 0.0:
            signals[i] = -0.25
    
    return signals

name = "4h_ATR_Regime_BB_Donchian_Volume"
timeframe = "4h"
leverage = 1.0