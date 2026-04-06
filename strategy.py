#!/usr/bin/env python3
"""
1d Donchian Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Donchian channel breakouts capture significant price moves, 
volume confirmation ensures breakout validity, and ADX filter avoids false signals in ranging markets.
Works in bull markets (breakouts to upside) and bear markets (breakouts to downside).
Target: 75-150 total trades over 4 years (19-38/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_adx_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for ADX filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        
        # Handle first element
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        dm_plus_sum = pd.Series(dm_plus).rolling(window=period, min_periods=period).sum().values
        dm_minus_sum = pd.Series(dm_minus).rolling(window=period, min_periods=period).sum().values
        
        # Directional Indicators
        plus_di = 100 * dm_plus_sum / tr_sum
        minus_di = 100 * dm_minus_sum / tr_sum
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        
        # Handle division by zero and NaN
        adx = np.where((plus_di + minus_di) == 0, 0, adx)
        return np.nan_to_num(adx, nan=0.0)
    
    adx_weekly = calculate_adx(high_weekly, low_weekly, close_weekly, 14)
    adx_weekly_aligned = align_htf_to_ltf(prices, df_weekly, adx_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(donchian_period, 20) + 14  # Donchian + volume MA + ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        strong_trend = adx_weekly_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.2 * 20-period average
        volume_confirm = volume[i] > (1.2 * vol_ma[i])
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR ADX weak OR stoploss
            if (close[i] <= donchian_low[i] or not strong_trend or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR ADX weak OR stoploss
            if (close[i] >= donchian_high[i] or not strong_trend or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume confirmation + strong trend
            long_breakout = close[i] > donchian_high[i]
            short_breakout = close[i] < donchian_low[i]
            
            if long_breakout and volume_confirm and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and volume_confirm and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals