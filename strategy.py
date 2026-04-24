#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 4h to target 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ADX(14) for trend strength (trending if ADX > 25) and 1d EMA50 for direction.
- Donchian channels: Upper = 20-period high, Lower = 20-period low (from prior 4h bars).
- Entry: Long when price breaks above Donchian Upper AND 1d ADX > 25 AND 1d close > EMA50 AND volume > 1.5 * volume MA(20).
         Short when price breaks below Donchian Lower AND 1d ADX > 25 AND 1d close < EMA50 AND volume > 1.5 * volume MA(20).
- Exit: ATR-based trailing stop - exit long when price < highest_high_since_entry - 3.0*ATR,
        exit short when price > lowest_low_since_entry + 3.0*ATR.
- Signal size: 0.25 discrete to minimize fee churn.
This strategy captures strong breakouts in established trends with volume confirmation,
using wider ATR trailing stops to let winners run while controlling risk. Works in both bull and bear markets
by requiring strong trend (ADX>25) and trading only in direction of 1d EMA50.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter and EMA50 direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ADX(14) for trend strength
    # ADX requires +DI, -DI, and DX calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM and -DM
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # +DI and -DI
    plus_di = 100 * dm_plus_smooth / atr_1d
    minus_di = 100 * dm_minus_smooth / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate Donchian channels (20-period) from prior 4h bars
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR(20) for trailing stop (4h)
    tr1_4h = np.abs(high[1:] - low[:-1])
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.max([tr1_4h[0], tr2_4h[0], tr3_4h[0]])], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50/ADX and Donchian/ATR/Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            trending = adx_aligned[i] > 25
            
            # Long: Price breaks above Donchian Upper AND trending AND bullish bias AND volume confirmed
            if (curr_close > donch_upper[i] and trending and 
                curr_close > ema_1d_aligned[i] and vol_confirmed):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: Price breaks below Donchian Lower AND trending AND bearish bias AND volume confirmed
            elif (curr_close < donch_lower[i] and trending and 
                  curr_close < ema_1d_aligned[i] and vol_confirmed):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit when price < highest_high - 3.0*ATR
            if curr_close < highest_since_entry - 3.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 3.0*ATR
            if curr_close > lowest_since_entry + 3.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADX_EMA50_Trend_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0