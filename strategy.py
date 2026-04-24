#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for trend filter (price above/below EMA34) and volume spike detection.
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.5x 20-period MA.
         Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.5x 20-period MA.
- Exit: Opposite Donchian breakout (price crosses Donchian(20) mid-line) OR trend filter fails.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in trend following.
- 1d EMA34 filter ensures we only trade in the direction of the higher timeframe trend.
- Volume confirmation avoids false breakouts during low participation periods.
- Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).
- Estimated trades: ~80 total over 4 years (~20/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period):
    """Calculate Donchian channels: upper, lower, middle."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2
    return upper, lower, middle

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6h Donchian(20) channels
    donch_hi, donch_lo, donch_mid = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for Donchian/EMA/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or np.isnan(donch_mid[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_1d_aligned[i]
        
        # Exit conditions: opposite Donchian breakout OR trend filter fails
        if position != 0:
            # Exit long: price falls below Donchian middle OR price falls below 1d EMA34
            if position == 1:
                if curr_close < donch_mid[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above Donchian middle OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > donch_mid[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Volume confirmation: current volume > 1.5x 1d volume MA
        volume_confirmed = curr_volume > 1.5 * curr_vol_ma
        
        # Entry conditions: Donchian breakout with trend and volume confirmation
        if position == 0:
            # Long: price breaks above Donchian high AND above 1d EMA34 AND volume confirmed
            if curr_close > donch_hi[i] and curr_close > ema34_1d_aligned[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1d EMA34 AND volume confirmed
            elif curr_close < donch_lo[i] and curr_close < ema34_1d_aligned[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0