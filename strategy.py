#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Uses Donchian channels from 4h chart for breakout structure, 1d ATR(14) for volatility regime filter,
# and volume spike for confirmation. Designed for 20-40 trades/year to minimize fee drag.
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns.
# The 1d ATR(14) regime filter avoids whipsaw in low volatility ranging markets and
# only allows breakouts during elevated volatility periods which tend to sustain trends.

name = "4h_Donchian20_1dATR14_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on 1d timeframe from prior completed 1d bar
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align length with 1d arrays
    
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_shifted = np.roll(atr14_1d, 1)
    atr14_1d_shifted[0] = np.nan
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d_shifted)
    
    # Calculate ATR percentile rank over 50 periods for regime filter
    # High volatility regime: ATR > 60th percentile of last 50 periods
    atr_percentile = pd.Series(atr14_1d_aligned).rolling(window=50, min_periods=20).apply(
        lambda x: np.nanpercentile(x, 60) if len(x) >= 20 else np.nan, raw=True
    ).values
    
    # Calculate Donchian(20) channels from prior completed 4h bar
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Shift by 1 to use only completed bars (no look-ahead)
    donchian_upper_shifted = np.roll(donchian_upper, 1)
    donchian_lower_shifted = np.roll(donchian_lower, 1)
    donchian_upper_shifted[0] = np.nan
    donchian_lower_shifted[0] = np.nan
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr_percentile[i]) or
            np.isnan(donchian_upper_shifted[i]) or
            np.isnan(donchian_lower_shifted[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime filter: only trade when ATR is above 60th percentile (elevated volatility)
        vol_regime = atr14_1d_aligned[i] > atr_percentile[i]
        
        if position == 0:
            # Long conditions: price breaks above Donchian Upper AND volume spike AND high volatility regime
            if close[i] > donchian_upper_shifted[i] and volume[i] > (2.0 * vol_ema_20[i]) and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian Lower AND volume spike AND high volatility regime
            elif close[i] < donchian_lower_shifted[i] and volume[i] > (2.0 * vol_ema_20[i]) and vol_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian Lower OR volatility regime breaks down
            if close[i] < donchian_lower_shifted[i] or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian Upper OR volatility regime breaks down
            if close[i] > donchian_upper_shifted[i] or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals