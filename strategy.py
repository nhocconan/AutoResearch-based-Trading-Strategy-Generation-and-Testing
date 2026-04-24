#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation + ATR(14) stoploss.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend direction.
- Donchian: Upper/lower 20-period channel on 4h data.
- Entry: Long when close > Upper20 AND price > 12h EMA50 AND volume > 1.5 * 20-period average volume.
         Short when close < Lower20 AND price < 12h EMA50 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout (close < Lower20 for long exit, close > Upper20 for short exit) OR ATR stoploss.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture momentum; EMA50 filter avoids counter-trend trades; volume confirms strength.
- Works in bull markets (upward breakouts with trend) and bear markets (downward breakouts with trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]  # first bar
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    if n < 20:
        return np.zeros(n)
    
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h volume average for confirmation (20-period)
    if n < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR for stoploss
    atr_period = 14
    if n < atr_period:
        return np.zeros(n)
    
    atr_val = atr(high, low, close, atr_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_val[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        long_breakout = curr_close > highest_20[i]
        short_breakout = curr_close < lowest_20[i]
        
        # Trend filter: price relative to 12h EMA50
        price_above_ema = curr_close > ema50_12h_aligned[i]
        price_below_ema = curr_close < ema50_12h_aligned[i]
        
        # Stoploss levels
        if position == 1:
            # Long stoploss: entry price - 2.0 * ATR
            stop_loss = entry_price - 2.0 * atr_val[i]
            if curr_low <= stop_loss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:
            # Short stoploss: entry price + 2.0 * ATR
            stop_loss = entry_price + 2.0 * atr_val[i]
            if curr_high >= stop_loss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below lower Donchian channel
            if position == 1 and curr_close < lowest_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
            # Exit short: price breaks above upper Donchian channel
            elif position == -1 and curr_close > highest_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Long: breakout above upper channel AND price > EMA50 AND volume confirmation
            long_condition = long_breakout and price_above_ema and volume_confirm
            
            # Short: breakout below lower channel AND price < EMA50 AND volume confirmation
            short_condition = short_breakout and price_below_ema and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0