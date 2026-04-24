#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ADX regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation (above 20-period SMA) and ADX regime (ADX > 25 for trending).
- Entry: Long when price breaks above Donchian(20) high AND 1d volume > 1d volume SMA(20) AND 1d ADX > 25.
         Short when price breaks below Donchian(20) low AND 1d volume > 1d volume SMA(20) AND 1d ADX > 25.
- Exit: Opposite Donchian breakout OR ADX falls below 20 (regime change to ranging).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in trend following.
- Volume confirmation ensures breakouts have participation.
- ADX filter avoids whipsaws in ranging markets (ADX < 20) and only trades in trending regimes (ADX > 25).
- Works in bull markets (buy breakouts in uptrends) and bear markets (sell breakdowns in downtrends).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period):
    """Calculate Donchian channels: upper=rolling max(high), lower=rolling min(low)."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
    return upper, lower

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high_series - high_series.shift(1)
    dm_minus = low_series.shift(1) - low_series
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_plus_smooth = dm_plus.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    dm_minus_smooth = dm_minus.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # Directional Index
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # ADX
    adx_values = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx_values.values, di_plus.values, di_minus.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    upper_12h, lower_12h = donchian_channels(high, low, 20)
    
    # Calculate 1d HTF indicators ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d volume SMA(20) for confirmation
    volume_sma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_1d)
    
    # 1d ADX(25) for regime filter
    adx_1d, _, _ = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 25)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for Donchian/ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or
            np.isnan(volume_sma_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR ADX falls below 20 (regime change)
        if position != 0:
            # Exit long: price breaks below Donchian low OR ADX < 20 (ranging)
            if position == 1:
                if curr_close < lower_12h[i] or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR ADX < 20 (ranging)
            elif position == -1:
                if curr_close > upper_12h[i] or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and ADX > 25 (trending)
        if position == 0:
            # Long: price breaks above Donchian high AND volume > volume SMA AND ADX > 25
            if (curr_close > upper_12h[i] and 
                curr_volume > volume_sma_1d_aligned[i] and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume > volume SMA AND ADX > 25
            elif (curr_close < lower_12h[i] and 
                  curr_volume > volume_sma_1d_aligned[i] and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolume_ADX_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0