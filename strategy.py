#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ATR filter and volume confirmation.
# Donchian breakout captures strong directional moves.
# 1w ATR filter ensures we only trade when volatility is sufficient (avoid choppy markets).
# Volume confirmation (>1.5x 20-period average) adds conviction to breakouts.
# Works in bull markets (upward breakouts) and bear markets (downward breakouts).
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
name = "1d_Donchian20_1wATR_Filter_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ATR filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w ATR (14-period)
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w.shift(1))
    tr3 = np.abs(low_1w - close_1w.shift(1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(atr_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: only trade when volatility is sufficient (> 0.5 * price)
        min_atr_threshold = 0.5 * close[i] * 0.01  # 0.5% of price as minimum ATR
        sufficient_volatility = atr_1w_aligned[i] > min_atr_threshold
        
        if position == 0:
            # Long: price breaks above upper Donchian band + volume confirmation + sufficient volatility
            if close[i] > high_roll[i-1] and volume_confirmed[i] and sufficient_volatility:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band + volume confirmation + sufficient volatility
            elif close[i] < low_roll[i-1] and volume_confirmed[i] and sufficient_volatility:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian band
            if close[i] < low_roll[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian band
            if close[i] > high_roll[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals