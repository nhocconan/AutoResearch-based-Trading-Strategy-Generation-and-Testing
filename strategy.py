#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with volume confirmation and 1-day ATR filter.
# Works in bull (breakouts continue) and bear (mean reversion at lower band in range) via price action at key levels.
# Target: 15-40 trades/year (60-160 total over 4 years) to avoid fee drag.
name = "4h_Donchian20_Volume_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR (14) for volatility filter
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to 4h (wait for daily close)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_20 = high_20[i]
        lower_20 = low_20[i]
        atr_val = atr_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume confirmation and sufficient volatility
            if close_val > upper_20 and vol_filter and (atr_val > 0):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume confirmation and sufficient volatility
            elif close_val < lower_20 and vol_filter and (atr_val > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower Donchian or ATR drops too low (low volatility)
            if close_val < lower_20 or (atr_val < 0.5 * atr_aligned[i-1] if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper Donchian or ATR drops too low (low volatility)
            if close_val > upper_20 or (atr_val < 0.5 * atr_aligned[i-1] if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals