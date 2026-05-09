#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation
# Uses 4h Donchian(20) breakout with 1d ATR(14) for volatility regime filter and volume spike confirmation
# Long when: price > Donchian upper + ATR(14) > 1.5x 50-period average + volume > 1.5x 20-period average
# Short when: price < Donchian lower + ATR(14) > 1.5x 50-period average + volume > 1.5x 20-period average
# Exit when: price crosses Donchian midline OR ATR volatility drops below threshold
# Position size: 0.25 to limit drawdown. Target: 30-50 trades/year.
# Designed to capture strong breakouts in volatile markets while avoiding choppy periods.

name = "4h_Donchian20_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14) using Wilder's smoothing
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR volatility filter: current ATR > 1.5x 50-period average
    atr_ma = pd.Series(atr_14).rolling(window=50, min_periods=50).mean()
    atr_high_vol = atr_14 > (1.5 * atr_ma.values)
    
    # Align daily ATR filter to 4h timeframe
    atr_high_vol_aligned = align_htf_to_ltf(prices, df_1d, atr_high_vol)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(lookback - 1, len(high)):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Donchian midline for exit
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_high_vol_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Donchian upper + high volatility + volume spike
            if (close[i] > donchian_high[i] and 
                atr_high_vol_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Donchian lower + high volatility + volume spike
            elif (close[i] < donchian_low[i] and 
                  atr_high_vol_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below midline OR volatility drops
            if (close[i] < donchian_mid[i]) or (not atr_high_vol_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above midline OR volatility drops
            if (close[i] > donchian_mid[i]) or (not atr_high_vol_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals