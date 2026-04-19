#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1-day volatility filter and volume confirmation.
# Long when: price breaks above 20-period high, ATR ratio > 1.2 (vol expansion), volume > 1.3x average
# Short when: price breaks below 20-period low, ATR ratio > 1.2, volume > 1.3x average
# Exit when price crosses the opposite Donchian boundary (e.g., long exits at 20-period low).
# ATR ratio filters for volatility expansion regimes, effective in both bull and bear markets.
# Target: 15-25 trades/year per symbol. Uses volatility-based breakouts to capture trends.
name = "12h_Donchian20_VolATR_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for ATR (volatility filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on daily data (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr = np.concatenate([[np.nan] * 14, atr[14:]])  # align with original length
    
    # ATR ratio: current ATR / 50-period average (volatility expansion filter)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma_50 > 0, atr / atr_ma_50, 1.0)
    
    # Align ATR ratio to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channels (20-period) on 12h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_ratio_val = atr_ratio_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above 20-period high, vol expansion, volume confirmation
            if price > high_max_20[i] and atr_ratio_val > 1.2 and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-period low, vol expansion, volume confirmation
            elif price < low_min_20[i] and atr_ratio_val > 1.2 and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-period low
            if price < low_min_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-period high
            if price > high_max_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals