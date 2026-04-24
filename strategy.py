#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume spike confirmation, and ATR-based stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA50 trend filter to capture major trend direction.
- Donchian(20): Identifies breakouts from 20-period price channels.
- Entry: Long when price breaks above 20-period high AND price > 1d EMA50 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below 20-period low AND price < 1d EMA50 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Donchian breakout OR ATR-based stoploss (2.5 * ATR from entry).
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- Donchian breakouts capture momentum, effective in both trending and ranging markets with filters.
- 1d EMA50 provides strong long-term trend filter to avoid counter-trend trades during major moves.
- Volume spike confirmation ensures breakouts have participation, reducing false signals.
- ATR stoploss manages risk during adverse moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period=14):
    """Calculate Average True Range."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.ewm(span=period, adjust=False, min_periods=period).mean()
    return atr_values.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d, additional_delay_bars=1)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for stoploss (14-period)
    atr_values = atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for Donchian and ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Update ATR-based stoploss
        if position == 1:  # Long position
            stop_price = entry_price - 2.5 * atr_values[i]
            # Exit conditions: opposite Donchian breakout OR stoploss hit
            if curr_low < lowest_low[i] or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            stop_price = entry_price + 2.5 * atr_values[i]
            # Exit conditions: opposite Donchian breakout OR stoploss hit
            if curr_high > highest_high[i] or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 2.0 * 20-period average volume
            vol_ma_20_current = vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else 0
            volume_confirmed = curr_volume > 2.0 * vol_ma_20_current
            
            # Long: Price breaks above 20-period high AND price > 1d EMA50 AND volume confirmed
            if curr_high > highest_high[i] and curr_close > ema50_1d_aligned[i] and volume_confirmed:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short: Price breaks below 20-period low AND price < 1d EMA50 AND volume confirmed
            elif curr_low < lowest_low[i] and curr_close < ema50_1d_aligned[i] and volume_confirmed:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_TrendFilter_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0