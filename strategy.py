#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter (price above/below daily EMA34).
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.5 * 20-period SMA volume.
         Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.5 * 20-period SMA volume.
- Exit: Opposite Donchian breakout (price crosses midline) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian channels provide clear breakout levels with built-in trend following.
- Daily EMA34 filter ensures trades align with higher timeframe trend, reducing whipsaws.
- Volume spike confirmation adds conviction to breakouts, filtering weak moves.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian(20) channels on 4h
    lookback = 20
    upper_channel = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_channel = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Volume confirmation: volume > 1.5 * 20-period SMA volume
    vol_sma = sma(volume, 20)
    volume_spike = volume > (1.5 * vol_sma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20)  # Need sufficient data for Donchian and volume SMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(middle_channel[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below middle channel OR price falls below 1d EMA34
            if position == 1:
                if curr_close < middle_channel[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above middle channel OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > middle_channel[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + volume spike + 1d EMA34 trend alignment
        if position == 0:
            # Long: price breaks above upper channel AND volume spike AND bullish 1d trend
            if curr_high > upper_channel[i] and volume_spike[i] and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND volume spike AND bearish 1d trend
            elif curr_low < lower_channel[i] and volume_spike[i] and curr_close < ema34_1d_aligned[i]:
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