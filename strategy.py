#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1-day ATR filter and volume confirmation.
# Long when: price breaks above Donchian(20) high, 1d ATR > 20-period mean ATR, volume > 1.5x 20-period average
# Short when: price breaks below Donchian(20) low, 1d ATR > 20-period mean ATR, volume > 1.5x 20-period average
# Exit when price returns to opposite Donchian level or ATR condition fails.
# Donchian captures breakouts, ATR filter ensures sufficient volatility, volume confirms conviction.
# Target: 20-40 trades/year per symbol. Works in trending markets (bull and bear).
name = "4h_Donchian20_ATR1d_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_1d_mean = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_1d_mean_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_mean)
    
    # Donchian channels on 4h data (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_1d_mean_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        atr_val = atr_1d_aligned[i]
        atr_mean = atr_1d_mean_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high, ATR condition met, volume spike
            if (price > upper and atr_val > atr_mean and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low, ATR condition met, volume spike
            elif (price < lower and atr_val > atr_mean and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below Donchian low or ATR condition fails
            if price < lower or atr_val <= atr_mean:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above Donchian high or ATR condition fails
            if price > upper or atr_val <= atr_mean:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals