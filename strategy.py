#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR-based regime filter (high volatility = trend follow, low volatility = avoid).
- Entry: Long when price breaks above Donchian(20) high AND 1d ATR ratio > 1.2 AND volume > 1.5x 20-period average.
         Short when price breaks below Donchian(20) low AND 1d ATR ratio > 1.2 AND volume > 1.5x 20-period average.
- Exit: Opposite Donchian breakout OR ATR ratio < 0.8 (low volatility regime).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in trend following.
- ATR regime filter avoids ranging markets where breakouts fail.
- Volume confirmation ensures breakouts have participation.
- Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
- Estimated trades: ~100 total over 4 years (~25/year) based on filtered breakout frequency.
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

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_ma_1d = sma(atr_1d, 10)  # 10-period average of ATR
    atr_ratio_1d = atr_1d / atr_ma_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d, additional_delay_bars=0)
    
    # Donchian channels on 4h (20-period)
    donch_high, donch_low = donchian_channels(high, low, 20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = sma(volume, 20)
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for Donchian/ATR/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR low volatility regime
        if position != 0:
            # Exit long: price breaks below Donchian low OR ATR ratio < 0.8 (low vol)
            if position == 1:
                if curr_close < donch_low[i] or atr_ratio_1d_aligned[i] < 0.8:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR ATR ratio < 0.8 (low vol)
            elif position == -1:
                if curr_close > donch_high[i] or atr_ratio_1d_aligned[i] < 0.8:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + high volatility regime + volume confirmation
        if position == 0:
            # Long: price breaks above Donchian high AND high volatility AND volume confirmation
            if curr_close > donch_high[i] and atr_ratio_1d_aligned[i] > 1.2 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND high volatility AND volume confirmation
            elif curr_close < donch_low[i] and atr_ratio_1d_aligned[i] > 1.2 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0