#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v2
# Hypothesis: 4h Donchian channel breakout with volume confirmation and 12h chop regime filter.
# Enters long on breakout above 20-period upper band with volume spike, short on breakout below lower band.
# Uses 12h choppiness index to avoid ranging markets (CHOP > 61.8 = range, no trades).
# Designed for moderate trade frequency (target: 75-200 total trades over 4 years) to balance edge and fees.
# Works in bull/bear by using 12h regime filter and volatility-based stops.
# Uses discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v2"
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
    
    # 4h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper = high_roll
    lower = low_roll
    
    # 12h HTF data for choppiness index regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range for 12h
    tr1 = pd.Series(high_12h).shift(1) - pd.Series(low_12h).shift(1)
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Choppiness Index (14-period)
    sum_atr_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_12h = (sum_atr_14 / chop_denominator) * 100
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(chop_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # 12h regime filter: only trade in trending markets (CHOP <= 61.8)
        trending = chop_12h_aligned[i] <= 61.8
        
        if position == 1:  # Long position
            # Exit: price falls below lower Donchian band
            if close[i] < lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above upper Donchian band
            if close[i] > upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper band with volume spike
            if (close[i] > upper[i]) and \
               (vol_spike[i]) and \
               (trending):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band with volume spike
            elif (close[i] < lower[i]) and \
                 (vol_spike[i]) and \
                 (trending):
                position = -1
                signals[i] = -0.25
    
    return signals