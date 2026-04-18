#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with volume confirmation and ATR volatility filter.
# Captures breakouts in trending markets and avoids false breakouts in low volatility.
# Works in both bull and bear: breakouts continue in bull, mean-reversion at channel bounds in bear.
# Target: 20-150 total trades over 4 years (5-37/year) to minimize fee drag.

name = "12h_Donchian20_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for ATR and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channel (20-period) on 12h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR (14) for volatility filter
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume average on 12h data
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily ATR to 12h (wait for daily close)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = high_max[i]
        lower = low_min[i]
        atr_val = atr_aligned[i]
        vol_filter = volume[i] > (1.5 * vol_ma_20[i])
        
        if position == 0:
            # Long: break above upper band with volume and volatility
            if close_val > upper and vol_filter and (atr_val > 0):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume and volatility
            elif close_val < lower and vol_filter and (atr_val > 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower band or volatility drops
            if close_val < lower or (atr_val < 0.5 * atr_aligned[i-1] if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band or volatility drops
            if close_val > upper or (atr_val < 0.5 * atr_aligned[i-1] if i > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals