#!/usr/bin/env python3
# 4h_donchian_12h_volume_v1
# Hypothesis: 4h strategy using Donchian(20) breakout with 12h volume confirmation and ATR filter.
# Enters long when price breaks above 20-period high with volume spike and ATR > 0, short when breaks below 20-period low.
# Uses discrete sizing (±0.30) to minimize fee churn. Target: 75-200 trades over 4 years.
# Works in bull/bear by using price channels as dynamic support/resistance.
# ATR filter ensures volatility is sufficient for breakout validity, reducing false signals.
# Added volume spike requirement to reduce false breakouts and lower trade frequency.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_12h_volume_v1"
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
    
    # Get 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Volume spike detection on 12h (20-period volume average)
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (vol_ma_20_12h * 2.0)  # Volume at least 2x average
    
    # Align volume spike to 4h timeframe (completed 12h candle only)
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Donchian channels (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR filter for volatility (14-period ATR on 4h)
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
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below 20-period low
            if close[i] < lowest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price rises above 20-period high
            if close[i] > highest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long: price breaks above 20-period high with volume spike and ATR filter
            if (close[i] > highest_20[i]) and vol_spike_aligned[i] and atr_filter[i]:
                position = 1
                signals[i] = 0.30
            # Enter short: price breaks below 20-period low with volume spike and ATR filter
            elif (close[i] < lowest_20[i]) and vol_spike_aligned[i] and atr_filter[i]:
                position = -1
                signals[i] = -0.30
    
    return signals