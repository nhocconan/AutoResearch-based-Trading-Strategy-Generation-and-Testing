#!/usr/bin/env python3
# 12h_donchian_1d_volume_chop_v1
# Hypothesis: 12h strategy using Donchian(20) breakout from 1d timeframe with volume confirmation and 1d choppiness regime filter.
# Enters long when price breaks above 1d Donchian upper channel with volume spike, short when breaks below lower channel.
# Uses 1d choppiness index to avoid ranging markets (CHOP > 61.8 = range, no trades).
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to avoid fee drag.
# Works in bull/bear by using 1d regime filter and Donchian channels as dynamic support/resistance.
# Uses discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Donchian channels and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d
    # Upper = MAX(high, 20), Lower = MIN(low, 20)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (completed 1d candle only)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 1d choppiness index (14-period)
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # SUM(ATR, 14) / (MAX(HIGH,14) - MIN(LOW,14)) * 100
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_1d = (sum_atr_14 / chop_denominator) * 100
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume spike detection (20-period volume average on 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Daily regime filter: only trade in trending markets (CHOP <= 61.8)
        trending = chop_1d_aligned[i] <= 61.8
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower channel
            if close[i] < donch_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper channel
            if close[i] > donch_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper channel with volume spike
            if (close[i] > donch_high_aligned[i]) and \
               (vol_spike[i]) and \
               (trending):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower channel with volume spike
            elif (close[i] < donch_low_aligned[i]) and \
                 (vol_spike[i]) and \
                 (trending):
                position = -1
                signals[i] = -0.25
    
    return signals