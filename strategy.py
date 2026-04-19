#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day ATR filter and volume confirmation.
# Long when: Price breaks above Donchian upper channel, 1D ATR > 1.5x 20-period average, volume spike
# Short when: Price breaks below Donchian lower channel, 1D ATR > 1.5x 20-period average, volume spike
# Exit when: Price crosses back through the 20-period middle band
# Donchian provides trend-following structure, ATR filters for volatility expansion, volume confirms breakout.
# Target: 20-30 trades/year per symbol. Works in trending markets (bull/bear) by capturing breakouts.
name = "4h_Donchian20_ATR_Volume"
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
    
    # 1-day data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period ATR on daily data
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period ATR average for filter
    atr_ma_20 = pd.Series(atr_20).rolling(window=20, min_periods=20).mean().values
    
    # Align 1D ATR data to 4H timeframe
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # 20-period Donchian channels on 4H data
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll_max + low_roll_min) / 2.0
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for ATR and Donchian calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or 
            np.isnan(atr_20_aligned[i]) or np.isnan(atr_ma_20_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_roll_max[i]
        lower = low_roll_min[i]
        mid = donchian_mid[i]
        atr = atr_20_aligned[i]
        atr_ma = atr_ma_20_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above upper channel, ATR expansion, volume spike
            if (price > upper and close[i-1] <= upper and 
                atr > 1.5 * atr_ma and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below lower channel, ATR expansion, volume spike
            elif (price < lower and close[i-1] >= lower and 
                  atr > 1.5 * atr_ma and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below middle band
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above middle band
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals