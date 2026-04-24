#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ATR regime filter + volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR-based volatility regime (high volatility = trend follow, low volatility = avoid).
- Entry: Long when price breaks above Donchian(20) high AND 1d ATR ratio > 1.2 AND volume > 1.5x 20-period average.
         Short when price breaks below Donchian(20) low AND 1d ATR ratio > 1.2 AND volume > 1.5x 20-period average.
- Exit: Opposite Donchian breakout OR ATR regime shifts to low volatility (ATR ratio < 0.8).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear trend-following structure with defined risk.
- ATR regime filter avoids whipsaws in low-volatility ranging markets.
- Volume confirmation ensures breakouts have conviction.
- Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Set first TR to high-low to avoid NaN from roll
    tr.iloc[0] = high[0] - low[0]
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1d ATR for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_ma_1d = pd.Series(atr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d  # Current ATR vs its recent average
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for Donchian/ATR/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Exit conditions: opposite Donchian breakout OR ATR regime shifts to low volatility
        if position != 0:
            # Exit long: price breaks below Donchian low OR ATR regime turns low volatility
            if position == 1:
                if curr_low <= donchian_low[i] or atr_ratio_1d_aligned[i] < 0.8:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR ATR regime turns low volatility
            elif position == -1:
                if curr_high >= donchian_high[i] or atr_ratio_1d_aligned[i] < 0.8:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + high volatility regime + volume spike
        if position == 0:
            # Long: price breaks above Donchian high AND high volatility regime AND volume spike
            if curr_high >= donchian_high[i] and atr_ratio_1d_aligned[i] > 1.2 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND high volatility regime AND volume spike
            elif curr_low <= donchian_low[i] and atr_ratio_1d_aligned[i] > 1.2 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0