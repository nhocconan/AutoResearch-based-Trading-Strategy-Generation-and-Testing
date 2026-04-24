#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volume spike filter.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA50 for trend direction (bullish if price > EMA50, bearish if price < EMA50).
- Volume: Current 4h volume > 2.0 * 20-period ATR-scaled volume MA to filter low-momentum breakouts.
- Entry: Long when price breaks above 20-period high AND 1d EMA50 bullish AND volume spike.
         Short when price breaks below 20-period low AND 1d EMA50 bearish AND volume spike.
- Exit: Opposite Donchian level (20-period low for long, 20-period high for short) or ATR trailing stop (2.5 * ATR).
- Signal size: 0.30 discrete to balance return and drawdown.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian channels provide objective breakout levels. Combined with trend and momentum filters,
this avoids false breakouts and works in both bull and bear markets by only taking trades
in the direction of the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss and volume filter
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR-scaled volume MA on 1d (ATR(14) * volume)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close_1d = df_1d['close'].values
    tr1_1d = df_1d_high - df_1d_low
    tr2_1d = np.abs(np.roll(df_1d_high, 1) - df_1d_close_1d)
    tr3_1d = np.abs(np.roll(df_1d_low, 1) - df_1d_close_1d)
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    vol_scaled_1d = df_1d['volume'].values * atr_1d
    vol_ma_scaled_1d = pd.Series(vol_scaled_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_scaled_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_scaled_1d)
    
    # Volume filter: current 4h ATR-scaled volume > 2.0 * 20-period 1d ATR-scaled volume MA
    vol_scaled = volume * atr
    volume_filter = vol_scaled > (2.0 * vol_ma_scaled_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough bars for EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            np.isnan(atr[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_1d_aligned[i]
        vol_filt = volume_filter[i]
        atr_val = atr[i]
        
        if position == 0:
            # Check for entry signals with volume filter
            if vol_filt:
                # Bullish: price breaks above 20-period high AND 1d EMA50 bullish (price > EMA)
                if curr_high > highest_20[i] and curr_close > ema_val:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_high
                # Bearish: price breaks below 20-period low AND 1d EMA50 bearish (price < EMA)
                elif curr_low < lowest_20[i] and curr_close < ema_val:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
                    lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Long exit: price breaks below 20-period low OR ATR trailing stop hit
            if curr_low < lowest_20[i] or curr_close < (highest_since_entry - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Short exit: price breaks above 20-period high OR ATR trailing stop hit
            if curr_high > highest_20[i] or curr_close > (lowest_since_entry + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_ATRVolumeFilter_v1"
timeframe = "4h"
leverage = 1.0