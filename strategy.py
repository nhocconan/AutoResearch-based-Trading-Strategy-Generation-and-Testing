#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with Donchian channel breakout on 1d timeframe.
# Uses 1d Donchian(20) high/low as breakout levels. Entry when 12h close breaks
# above/below the 1d Donchian bands with volume confirmation. Volatility filter
# ensures breakouts occur during sufficient volatility. Works in both bull and
# bear markets by capturing breakouts in either direction. Target: 50-150 total
# trades over 4 years (12-37/year).
name = "12h_1d_Donchian20_Volume_VolatilityFilter"
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
    
    # Get 1d data for Donchian calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels on 1d timeframe (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    # Volatility filter: 1d ATR > 0.5 * 50-period average ATR (avoid low volatility)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_1d > (atr_ma_50 * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr_ma_50[i])):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long when price breaks above 1d Donchian high with volume and volatility
            if (close[i] > donchian_high_aligned[i] and volume_filter[i] and
                volatility_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 1d Donchian low with volume and volatility
            elif (close[i] < donchian_low_aligned[i] and volume_filter[i] and
                  volatility_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below 1d Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above 1d Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals