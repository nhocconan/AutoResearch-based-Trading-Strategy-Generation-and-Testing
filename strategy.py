#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d ATR-based volatility filter and volume confirmation
# Uses Donchian(20) for structure, 1d ATR(14) for volatility regime filter (low volatility = breakout prone),
# and volume spike for confirmation. Designed for 20-30 trades/year to minimize fee drag.
# Works in bull markets via upside breakouts and in bear markets via downside breakdowns.
# The 1d ATR filter ensures we only trade during low volatility regimes where breakouts are more reliable.

name = "4h_Donchian20_1dATR14_VolumeSpike_Filter"
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
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) for volatility filter from prior completed 1d bar
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_shifted = np.roll(atr14_1d, 1)
    atr14_1d_shifted[0] = np.nan
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d_shifted)
    
    # Calculate 20-period ATR percentile rank (20-day lookback) for regime filter
    atr_percentile = pd.Series(atr14_1d_aligned).rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) == 20 else np.nan, raw=False
    ).values
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(atr_percentile[i]) or 
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ATR percentile < 0.3 (low volatility environment)
        low_vol_regime = atr_percentile[i] < 0.3
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND low volatility regime AND volume spike
            if close[i] > highest_20[i] and low_vol_regime and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND low volatility regime AND volume spike
            elif close[i] < lowest_20[i] and low_vol_regime and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower band OR volatility regime changes to high
            if close[i] < lowest_20[i] or atr_percentile[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper band OR volatility regime changes to high
            if close[i] > highest_20[i] or atr_percentile[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals