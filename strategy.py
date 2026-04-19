#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with daily ATR filter and volume confirmation.
# Long when: Price breaks above 20-period high, daily ATR(14) > 10-period MA, volume > 1.5x 20-period average
# Short when: Price breaks below 20-period low, daily ATR(14) > 10-period MA, volume > 1.5x 20-period average
# Exit when: Price crosses back through the midline (average of 20-period high and low)
# Donchian channels provide clear breakout signals, ATR filter ensures volatility regime, volume confirms strength.
# Target: 20-30 trades/year per symbol. Works in trending markets (both bull and bear).
name = "12h_Donchian_Breakout_Volume_ATRFilter"
timeframe = "12h"
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
    
    # Calculate ATR(14) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 10-period MA of ATR
    atr_ma10 = pd.Series(atr14).rolling(window=10, min_periods=10).mean().values
    
    # Align 1D ATR data to 12H timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr_ma10_aligned = align_htf_to_ltf(prices, df_1d, atr_ma10)
    
    # Calculate 20-period Donchian channels on 12H data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr14_aligned[i]) or np.isnan(atr_ma10_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        atr14_val = atr14_aligned[i]
        atr_ma10_val = atr_ma10_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above 20-period high, ATR above MA, volume spike
            if (price > high_20_val and close[i-1] <= high_20_val and 
                atr14_val > atr_ma10_val and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below 20-period low, ATR above MA, volume spike
            elif (price < low_20_val and close[i-1] >= low_20_val and 
                  atr14_val > atr_ma10_val and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below midline
            if price < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above midline
            if price > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals