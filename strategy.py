#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ADX trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ADX trend filter (ADX > 25) and ATR-based volume spike detection.
- Entry: Long when price breaks above Donchian upper (20) AND ADX > 25 AND volume > 1.5 * avg_volume.
         Short when price breaks below Donchian lower (20) AND ADX > 25 AND volume > 1.5 * avg_volume.
- Exit: Opposite Donchian breakout OR price crosses Donchian midpoint.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide objective breakout levels that work in both bull and bear markets.
- ADX > 25 ensures we only trade in trending conditions, avoiding whipsaws in ranging markets.
- Volume confirmation (1.5x average) ensures breakouts have institutional participation.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period):
    """Calculate Donchian channels (upper, lower, middle)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    
    # Directional Movement
    up_move = pd.Series(high - np.roll(high, 1))
    down_move = pd.Series(np.roll(low, 1) - low)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_values = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx_values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper, donchian_lower, donchian_middle = donchian_channels(high, low, 20)
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d, additional_delay_bars=1)
    
    # Calculate 1d average volume for volume spike confirmation
    avg_volume_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for Donchian (20) + ADX (14) + volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses Donchian midpoint
        if position != 0:
            # Exit long: price breaks below Donchian lower OR price falls below Donchian middle
            if position == 1:
                if curr_close < donchian_lower[i] or curr_close < donchian_middle[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper OR price rises above Donchian middle
            elif position == -1:
                if curr_close > donchian_upper[i] or curr_close > donchian_middle[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above Donchian upper AND ADX > 25 AND volume > 1.5 * avg_volume
            if (curr_close > donchian_upper[i] and 
                adx_1d_aligned[i] > 25 and 
                curr_volume > 1.5 * avg_volume_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND ADX > 25 AND volume > 1.5 * avg_volume
            elif (curr_close < donchian_lower[i] and 
                  adx_1d_aligned[i] > 25 and 
                  curr_volume > 1.5 * avg_volume_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dADX_TrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0