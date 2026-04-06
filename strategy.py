#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d ADX Trend + Volume Spike + ATR Stop
Hypothesis: Uses daily ADX for trend strength filter, Donchian breakouts for entry,
and volume spikes for confirmation. ADX > 25 ensures we trade only in strong trends,
reducing whipsaw in ranging markets. Works in bull (breakouts with ADX>25) and bear
(breakdowns with ADX>25). Targets 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1dadx_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 1d ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and Directional Movement
    tr_1d = np.zeros(len(close_1d))
    dm_plus = np.zeros(len(close_1d))
    dm_minus = np.zeros(len(close_1d))
    
    if len(close_1d) >= 2:
        tr_1d[1:] = np.maximum(
            high_1d[1:] - low_1d[1:],
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
        dm_plus[1:] = np.where(
            (high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]),
            np.maximum(high_1d[1:] - high_1d[:-1], 0),
            0
        )
        dm_minus[1:] = np.where(
            (low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]),
            np.maximum(low_1d[:-1] - low_1d[1:], 0),
            0
        )
    
    # Smoothed TR, DM+, DM- (14-period Wilder's smoothing)
    tr_14 = np.full(len(close_1d), np.nan)
    dm_plus_14 = np.full(len(close_1d), np.nan)
    dm_minus_14 = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 14:
        tr_14[13] = np.sum(tr_1d[1:14])
        dm_plus_14[13] = np.sum(dm_plus[1:14])
        dm_minus_14[13] = np.sum(dm_minus[1:14])
        
        for i in range(14, len(close_1d)):
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr_1d[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = np.full(len(close_1d), np.nan)
    di_minus = np.full(len(close_1d), np.nan)
    dx = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 14:
        for i in range(14, len(close_1d)):
            if tr_14[i] != 0:
                di_plus[i] = 100 * dm_plus_14[i] / tr_14[i]
                di_minus[i] = 100 * dm_minus_14[i] / tr_14[i]
                if (di_plus[i] + di_minus[i]) != 0:
                    dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX: smoothed DX
    adx = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 28:  # Need 14 for DX + 14 for smoothing
        dx_valid = dx[~np.isnan(dx)]
        if len(dx_valid) >= 14:
            adx[27] = np.mean(dx_valid[:14])
            for i in range(28, len(close_1d)):
                if not np.isnan(dx[i]):
                    adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Trend strength: ADX > 25 indicates strong trend
    trend_strong = np.where(adx > 25, 1, 0)
    
    # Align to 4h timeframe
    trend_strong_aligned = align_htf_to_ltf(prices, df_1d, trend_strong)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(trend_strong_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR no strong trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                trend_strong_aligned[i] == 0 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR no strong trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                trend_strong_aligned[i] == 0 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + strong trend + volume spike
            # Minimum holding period: only allow new entry after 15 bars flat
            if bars_since_entry >= 15:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout with strong trend and volume
                if bull_breakout and trend_strong_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout with strong trend and volume
                elif bear_breakout and trend_strong_aligned[i] == 1 and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals