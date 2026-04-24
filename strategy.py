#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ADX trend filter and ATR volume confirmation.
- Long when price breaks above Donchian upper band (20-period high) AND 1d ADX > 25 (trending) AND volume > 1.5 * ATR(14) * close
- Short when price breaks below Donchian lower band (20-period low) AND 1d ADX > 25 (trending) AND volume > 1.5 * ATR(14) * close
- Exit on Donchian middle band (10-period median) cross for faster mean reversion in chop
- Uses 6h primary timeframe with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Donchian channels provide clear breakout levels that work in trending markets
- 1d ADX > 25 ensures we only trade in trending regimes, avoiding whipsaws in ranging markets
- ATR-scaled volume filter confirms breakout strength with institutional participation
- Designed for BTC/ETH with edge in trending markets (both bull and bear) while avoiding ranging chop
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) using previous period (no look-ahead)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    # Middle band = median of upper and lower (or 10-period EMA approximation)
    lookback = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(lookback, n):
        upper_band[i] = np.max(high[i-lookback:i])
        lower_band[i] = np.min(low[i-lookback:i])
    
    # Middle band as 10-period EMA of price (approximation for Donchian middle)
    middle_band = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = pd.Series(np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                                 np.maximum(high_1d - np.roll(high_1d, 1), 0), 0))
    dm_minus = pd.Series(np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                                  np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0))
    dm_plus.iloc[0] = np.nan
    dm_minus.iloc[0] = np.nan
    
    # Smoothed values
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(span=14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = dx.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate ATR(14) for dynamic volume threshold (6h timeframe)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.5 * ATR * close (volatility-adjusted)
    vol_threshold = 1.5 * atr * close
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band, trend strong (ADX > 25), volume confirmation
            if close[i] > upper_band[i] and adx_1d_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, trend strong (ADX > 25), volume confirmation
            elif close[i] < lower_band[i] and adx_1d_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian middle band (mean reversion in chop)
            if close[i] < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian middle band (mean reversion in chop)
            if close[i] > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dADX_ATRVolConfirm_v1"
timeframe = "6h"
leverage = 1.0