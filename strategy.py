#!/usr/bin/env python3
"""
1d Donchian Breakout + Volume Spike + ADX Filter
Hypothesis: Donchian breakouts with volume confirmation and ADX trend filter
capture significant moves in both bull and bear markets. Using 1d timeframe
reduces trade frequency to avoid fee drag, while 1w ADX provides trend strength
filter to avoid whipsaws. Volatility filter ensures trades occur during
expanded volatility periods. Target: 50-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Donchian calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load 1w data for ADX trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian Channel (20-period)
    donchian_period = 20
    donchian_high = pd.Series(high_1d).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low_1d).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ADX calculation (14-period) on 1w data
    adx_period = 14
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smoothed TR, DM+
    atr = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=adx_period, min_periods=adx_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=adx_period, min_periods=adx_period).mean().values
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Align indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 1d data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume spike (2x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_period, adx_period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_1d[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # ADX trend filter: only trade when trend is strong (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend weakens OR stoploss
            if (close[i] <= donchian_low_aligned[i] or not strong_trend or
                close[i] <= entry_price - 2.5 * atr_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend weakens OR stoploss
            if (close[i] >= donchian_high_aligned[i] or not strong_trend or
                close[i] >= entry_price + 2.5 * atr_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume spike + strong trend
            long_breakout = close[i] > donchian_high_aligned[i]
            short_breakout = close[i] < donchian_low_aligned[i]
            
            if long_breakout and vol_spike[i] and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and vol_spike[i] and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals