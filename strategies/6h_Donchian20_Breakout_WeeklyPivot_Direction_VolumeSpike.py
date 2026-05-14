#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike
Hypothesis: On 6h timeframe, Donchian(20) breakout in the direction of weekly pivot trend (price > weekly pivot = bullish, price < weekly pivot = bearish) with volume confirmation (>1.5x 20-period MA) captures high-probability trend continuation moves. Weekly pivot acts as dynamic support/resistance and regime filter. Discrete position sizing (±0.25) and ATR-based trailing stop (2.5x) for exits. Targets 12-30 trades/year by requiring weekly regime alignment, volume confirmation, and Donchian breakout structure—designed to work in both bull (breakouts above weekly pivot) and bear (breakdowns below weekly pivot) markets with BTC/ETH edge from institutional pivot levels and volume-confirmed breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly pivot: (H+L+C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Daily data for Donchian channel calculation (more responsive than weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian(20) from daily: upper = max(high, 20), lower = min(low, 20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling max/min with proper min_periods
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # 6h ATR(14) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = tr_6h.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_6h_values = atr_6h.values
    
    # Volume spike filter: volume > 1.5 * 20-period MA on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of Donchian (20), ATR (14), volume MA (20), weekly pivot needs 1 week
    start_idx = max(20, 14, 20) + 48  # +48 to ensure 1 week of 6h data for weekly pivot
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        pivot_val = weekly_pivot_aligned[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(pivot_val) or np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Regime filter: bullish when price > weekly pivot, bearish when price < weekly pivot
        regime_bullish = close_val > pivot_val
        regime_bearish = close_val < pivot_val
        
        # Donchian breakout conditions: price breaks upper/lower with regime alignment + volume spike
        long_breakout = close_val > upper_val
        short_breakout = close_val < lower_val
        
        long_entry = regime_bullish and long_breakout and vol_spike
        short_entry = regime_bearish and short_breakout and vol_spike
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0