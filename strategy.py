#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter and ATR-based volume confirmation.
- Entry: Long when price breaks above 20-period Donchian high AND price > 1d EMA50 AND 12h volume > 1.5 * 20-period average volume.
         Short when price breaks below 20-period Donchian low AND price < 1d EMA50 AND 12h volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear breakout levels with built-in trend following.
- 1d EMA50 provides medium-term trend filter to avoid counter-trend trades.
- Volume confirmation ensures breakouts have participation, reducing false signals.
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channel(high, low, period):
    """Calculate Donchian Channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume confirmation threshold
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d, additional_delay_bars=1)
    
    # Calculate 12h Donchian Channel (20-period)
    dc_upper, dc_lower = donchian_channel(high, low, 20)
    
    # Calculate 12h volume moving average (20-period)
    if len(volume) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 20)  # Need sufficient data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_vol_ratio = curr_volume / (vol_ma_20[i] + 1e-10)  # Avoid division by zero
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 1d EMA50
            if position == 1:
                if curr_low < dc_lower[i] or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 1d EMA50
            elif position == -1:
                if curr_high > dc_upper[i] or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above Donchian high AND price > 1d EMA50 AND volume confirmation
            if curr_high > dc_upper[i] and curr_close > ema50_1d_aligned[i] and curr_vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < 1d EMA50 AND volume confirmation
            elif curr_low < dc_lower[i] and curr_close < ema50_1d_aligned[i] and curr_vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_TrendFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0