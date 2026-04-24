#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1d choppiness regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation and choppiness regime.
- Entry: Long when price breaks above Donchian(20) high AND 1d volume > 1.5 * 20-period average AND choppiness > 61.8 (range regime).
         Short when price breaks below Donchian(20) low AND 1d volume > 1.5 * 20-period average AND choppiness > 61.8.
- Exit: Opposite Donchian breakout OR choppiness < 38.2 (trend regime) to avoid whipsaw.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian channels provide clear breakout levels with built-in trend following.
- Volume spike confirms institutional participation.
- Choppiness regime filter ensures we only trade in ranging markets (mean reversion) where breakouts are more reliable.
- Works in bull markets (buy breakouts in range) and bear markets (sell breakdowns in range).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
    return upper, lower

def choppiness_index(high, low, close, period):
    """Calculate Choppiness Index: measures whether market is choppy (ranging) or trending."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of True Range over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    hh = high_series.rolling(window=period, min_periods=period).max()
    ll = low_series.rolling(window=period, min_periods=period).min()
    
    # Choppiness Index formula
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian(20) channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    upper_12h, lower_12h = donchian_channels(df_12h['high'].values, df_12h['low'].values, 20)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Calculate 1d volume spike and choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume: current volume > 1.5 * 20-period average
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 1d choppiness index
    chop_1d = choppiness_index(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Regime filters: chop > 61.8 = range (good for mean reversion breakouts), chop < 38.2 = trend (avoid)
    chop_range_1d = chop_1d_aligned > 61.8
    chop_trend_1d = chop_1d_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for Donchian/MA/chop
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR chop < 38.2 (trend regime)
        if position != 0:
            # Exit long: price breaks below lower Donchian OR chop < 38.2 (trend regime)
            if position == 1:
                if curr_close < lower_12h_aligned[i] or chop_trend_1d[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper Donchian OR chop < 38.2 (trend regime)
            elif position == -1:
                if curr_close > upper_12h_aligned[i] or chop_trend_1d[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + volume spike + range regime
        if position == 0:
            # Long: price breaks above upper Donchian AND volume spike AND range regime
            if curr_close > upper_12h_aligned[i] and volume_spike_1d_aligned[i] and chop_range_1d[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND volume spike AND range regime
            elif curr_close < lower_12h_aligned[i] and volume_spike_1d_aligned[i] and chop_range_1d[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ChopRange_v1"
timeframe = "12h"
leverage = 1.0