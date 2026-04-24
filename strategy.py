#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for trend filter (price above/below EMA34) and volume spike (current volume > 1.5 * 20-period average volume).
- Entry: Long when price breaks above Donchian(20) upper band AND price > 1d EMA34 AND volume spike.
         Short when price breaks below Donchian(20) lower band AND price < 1d EMA34 AND volume spike.
- Exit: Opposite Donchian breakout (price crosses opposite band) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in trend following.
- EMA34 filter ensures we only trade in the direction of the higher timeframe trend.
- Volume spike confirms breakout validity and reduces false signals.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~120 total over 4 years (~30/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for spike detection
    vol_avg_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d, additional_delay_bars=1)
    
    # Donchian(20) channels on 4h
    lookback = 20
    upper_band = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_band = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike condition on 4h: current volume > 1.5 * 20-period average volume
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback)  # Need sufficient data for Donchian and EMAs
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below lower band OR price falls below 1d EMA34
            if position == 1:
                if curr_close < lower_band[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper band OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > upper_band[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + trend filter + volume spike
        if position == 0:
            # Long: price breaks above upper band AND price > 1d EMA34 AND volume spike
            if curr_close > upper_band[i] and curr_close > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND price < 1d EMA34 AND volume spike
            elif curr_close < lower_band[i] and curr_close < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0