#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day ATR filter and volume confirmation.
# Long when: Price breaks above 20-period high, daily ATR < 10-day ATR mean, volume > 1.3x 20-period average
# Short when: Price breaks below 20-period low, daily ATR < 10-day ATR mean, volume > 1.3x 20-period average
# Exit when: Price crosses back through the 20-period midpoint
# ATR filter avoids breakouts during high volatility (false breakouts), volume confirms strength.
# Target: 20-30 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "4h_Donchian20_ATRFilter_Volume"
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
    
    # 1-day data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 4h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0
    
    # Calculate ATR on daily data (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_10 = pd.Series(atr_14).rolling(window=10, min_periods=10).mean().values
    
    # Align 1D data to 4H timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_10_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_10)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_10_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_max = high_max_20[i]
        low_min = low_min_20[i]
        midpoint = donchian_mid[i]
        atr = atr_14_aligned[i]
        atr_ma = atr_ma_10_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above 20-period high, low ATR regime, volume spike
            if (price > high_max and close[i-1] <= high_max and 
                atr < atr_ma and vol > 1.3 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below 20-period low, low ATR regime, volume spike
            elif (price < low_min and close[i-1] >= low_min and 
                  atr < atr_ma and vol > 1.3 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below midpoint
            if price < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above midpoint
            if price > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals