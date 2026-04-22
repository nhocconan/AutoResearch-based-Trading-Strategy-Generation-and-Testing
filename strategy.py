# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6H_Combined_RangeBreakout_Momentum
Hypothesis: Combines range breakout (Donchian) with momentum (RSI) and volume confirmation.
Works in bull (breakouts) and bear (mean reversion in range) via regime filter.
Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
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
    
    # Load 1d data for regime filter (trend vs range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX for regime: ADX > 25 = trending, ADX < 20 = ranging
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth(val, period):
        res = np.zeros_like(val)
        res[:period-1] = np.nan
        if len(val) >= period:
            res[period-1] = np.mean(val[:period])
            for i in range(period, len(val)):
                res[i] = (res[i-1] * (period-1) + val[i]) / period
        return res
    
    atr = smooth(tr, 14)
    dm_plus_smooth = smooth(dm_plus, 14)
    dm_minus_smooth = smooth(dm_minus, 14)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth(dx, 14)
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h Donchian channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # 6h RSI (14-period) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        if i < 14:
            avg_gain[i] = np.mean(gain[:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_avg_20 = np.zeros(n)
    for i in range(20, n):
        vol_avg_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entry
            if adx_aligned[i] > 25:  # Trending regime
                # Breakout with volume and momentum
                if (close[i] > highest_high[i] and volume[i] > 1.5 * vol_avg_20[i] and rsi[i] > 50):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] < lowest_low[i] and volume[i] > 1.5 * vol_avg_20[i] and rsi[i] < 50):
                    signals[i] = -0.25
                    position = -1
            else:  # Ranging regime (ADX < 25)
                # Mean reversion at Donchian bands
                if (close[i] < lowest_low[i] and volume[i] > 1.5 * vol_avg_20[i] and rsi[i] < 30):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] > highest_high[i] and volume[i] > 1.5 * vol_avg_20[i] and rsi[i] > 70):
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:  # Long
                # Exit on opposite touch or momentum loss
                if (close[i] < lowest_low[i] or rsi[i] < 40):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short
                # Exit on opposite touch or momentum loss
                if (close[i] > highest_high[i] or rsi[i] > 60):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Combined_RangeBreakout_Momentum"
timeframe = "6h"
leverage = 1.0