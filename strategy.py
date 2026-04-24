#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ATR filter + volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR-based volatility regime filter (only trade when ATR(14) > 1d ATR(50) percentile 60).
- Entry: Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar average volume.
         Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar average volume.
- Exit: Opposite Donchian breakout (price crosses midline) OR ATR volatility drops below percentile 40.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian channels provide clear breakout levels with built-in trend following.
- ATR filter avoids low-volatility choppy markets where breakouts fail.
- Volume confirmation ensures breakouts have conviction.
- Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
- Estimated trades: ~100 total over 4 years (~25/year) based on filtered breakout frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period):
    """Calculate Donchian channels: upper=highest(high, period), lower=lowest(low, period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]  # First value
    low_close[0] = high_low[0]   # First value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_vals = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_vals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR filter: ATR(14) vs ATR(50) percentile
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    atr14_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr50_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 50)
    
    # Calculate percentile rank of ATR14 relative to ATR50 (60% threshold for high vol regime)
    atr_ratio = atr14_1d / (atr50_1d + 1e-10)  # Avoid division by zero
    # Use rolling percentile: when ATR14 is above 60th percentile of ATR50, regime is favorable
    atr_percentile = pd.Series(atr_ratio).rolling(window=50, min_periods=20).apply(
        lambda x: np.percentile(x, 60) if len(x) == 50 else np.nan, raw=True
    ).values
    vol_regime_favorable = atr_ratio > atr_percentile  # High volatility regime
    
    vol_regime_favorable_aligned = align_htf_to_ltf(prices, df_1d, vol_regime_favorable, additional_delay_bars=1)
    
    # Donchian(20) on 4h
    donch_period = 20
    upper, lower = donchian_channels(high, low, donch_period)
    midline = (upper + lower) / 2
    
    # Volume confirmation: volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Donchian/ATR/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or
            np.isnan(vol_regime_favorable_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR volatility regime turns unfavorable
        if position != 0:
            # Exit long: price falls below midline OR volatility regime unfavorable
            if position == 1:
                if curr_close < midline[i] or not vol_regime_favorable_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above midline OR volatility regime unfavorable
            elif position == -1:
                if curr_close > midline[i] or not vol_regime_favorable_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + volume confirmation + favorable volatility regime
        if position == 0:
            # Long: price breaks above upper Donchian band AND volume confirms AND favorable vol regime
            if curr_close > upper[i] and volume_confirm[i] and vol_regime_favorable_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band AND volume confirms AND favorable vol regime
            elif curr_close < lower[i] and volume_confirm[i] and vol_regime_favorable_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATR_Volume_Regime_Breakout_v1"
timeframe = "4h"
leverage = 1.0