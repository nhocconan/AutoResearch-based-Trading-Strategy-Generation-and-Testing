#!/usr/bin/env python3
# 12h_donchian_1d_volume_atr_v1
# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and ATR filter.
# Enters long when price breaks above 1d Donchian upper (20) with volume spike and ATR > 0,
# short when price breaks below 1d Donchian lower (20) with volume spike and ATR > 0.
# Uses discrete position sizing (±0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull/bear via price channel structure.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_1d_volume_atr_v1"
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
    
    # Get 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper and lower (20-period)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (completed HTF candle only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume spike detection (20-period volume average on 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    # ATR filter for volatility (14-period ATR on 12h)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period: use only high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_filter = atr > 0  # Ensure ATR is valid (non-zero)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below 1d Donchian lower
            if close[i] < donchian_lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above 1d Donchian upper
            if close[i] > donchian_upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above 1d Donchian upper with volume spike and ATR filter
            if (close[i] > donchian_upper_aligned[i]) and vol_spike[i] and atr_filter[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below 1d Donchian lower with volume spike and ATR filter
            elif (close[i] < donchian_lower_aligned[i]) and vol_spike[i] and atr_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals