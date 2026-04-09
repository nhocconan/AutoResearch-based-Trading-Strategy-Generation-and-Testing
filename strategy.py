#!/usr/bin/env python3
# 4h_donchian_1d_volume_atr_v1
# Hypothesis: 4h Donchian channel breakout with daily volume confirmation and ATR filter.
# Enters long when price breaks above 20-period Donchian high with volume spike and ATR > 0,
# short when breaks below 20-period Donchian low. Uses discrete sizing (±0.30) to minimize fee churn.
# Target: 75-200 trades over 4 years (19-50/year). Works in bull/bear by using volatility-based breakouts.
# Volume spike reduces false breakouts, ATR ensures sufficient volatility for valid breakout.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_volume_atr_v1"
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
    
    # Get 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily volume spike detection (20-period volume average on 1d)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_20_1d * 2.0)  # Volume at least 2x daily average
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Daily ATR filter (14-period ATR on 1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]  # First period: use only high-low
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_filter_1d = atr_1d > 0  # Ensure ATR is valid (non-zero)
    atr_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_filter_1d)
    
    # 4h Donchian channel (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(atr_filter_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long: price breaks above Donchian high with daily volume spike and ATR filter
            if (close[i] > donchian_high[i]) and vol_spike_1d_aligned[i] and atr_filter_1d_aligned[i]:
                position = 1
                signals[i] = 0.30
            # Enter short: price breaks below Donchian low with daily volume spike and ATR filter
            elif (close[i] < donchian_low[i]) and vol_spike_1d_aligned[i] and atr_filter_1d_aligned[i]:
                position = -1
                signals[i] = -0.30
    
    return signals