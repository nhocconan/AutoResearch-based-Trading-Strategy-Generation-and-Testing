#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout + 1d volume spike + 1d ATR filter
# - Long when price breaks above 12h Donchian upper (20) + volume > 1.5x 20-period average + ATR(14) < 0.5 * price
# - Short when price breaks below 12h Donchian lower (20) + volume > 1.5x 20-period average + ATR(14) < 0.5 * price
# - Exit when price crosses back through Donchian midpoint or volatility expands (ATR > 0.7 * price)
# - Uses 1d ATR for volatility filter to avoid high-volatility periods
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels for 12h timeframe (20-period)
    high_max = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_max
    donchian_lower = low_min
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian levels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Load 1d data for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Current price and volume
    price = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian/volume/ATR warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or \
           np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        p = price[i]
        v = volume[i]
        vol_ma = vol_ma_1d_aligned[i]
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + volume spike + low volatility
            if p > donchian_upper_aligned[i] and v > 1.5 * vol_ma and atr < 0.5 * p:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + volume spike + low volatility
            elif p < donchian_lower_aligned[i] and v > 1.5 * vol_ma and atr < 0.5 * p:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian mid or volatility expands
            if p < donchian_mid_aligned[i] or atr > 0.7 * p:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian mid or volatility expands
            if p > donchian_mid_aligned[i] or atr > 0.7 * p:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_ATRFilter"
timeframe = "12h"
leverage = 1.0