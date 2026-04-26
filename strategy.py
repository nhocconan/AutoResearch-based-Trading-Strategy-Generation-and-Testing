#!/usr/bin/env python3
"""
6h_Donchian_20_WeeklyPivot_Volume_1dTrend_v1
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts aligned with weekly pivot direction (from 1w HTF) with volume confirmation and 1d EMA50 trend filter. Weekly pivot provides structural bias, Donchian gives breakout entry, volume confirms participation, and 1d trend avoids counter-trend trades. Designed for low frequency (12-37 trades/year) to minimize fee drag while capturing sustained moves in both bull and bear markets via trend alignment.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for weekly pivot (prior week OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot levels from prior week (HLC of prior 1w)
    wk_high = pd.Series(df_1w['high'].values).shift(1).values
    wk_low = pd.Series(df_1w['low'].values).shift(1).values
    wk_close = pd.Series(df_1w['close'].values).shift(1).values
    wk_pivot = (wk_high + wk_low + wk_close) / 3.0
    
    # Get 6h data for Donchian(20) and volume average
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Donchian(20) from prior 20 periods (6h bars)
    donch_high = pd.Series(df_6h['high'].values).shift(1).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_6h['low'].values).shift(1).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(df_6h['volume'].values).shift(1).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    wk_pivot_aligned = align_htf_to_ltf(prices, df_1w, wk_pivot)
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(50) 1d (50), weekly pivot (need 2 bars for shift), Donchian(20) (need 20+1)
    start_idx = max(50, 2, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(wk_pivot_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_val = volume[i]
        pivot_val = wk_pivot_aligned[i]
        dc_high = donch_high_aligned[i]
        dc_low = donch_low_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        # Pivot bias: above pivot = long bias, below pivot = short bias
        long_bias = close_val > pivot_val
        short_bias = close_val < pivot_val
        
        if position == 0:
            # Long: break above Donchian high with uptrend, volume, and pivot bias
            long_signal = (high_val > dc_high) and \
                          uptrend and \
                          volume_confirm and \
                          long_bias
            
            # Short: break below Donchian low with downtrend, volume, and pivot bias
            short_signal = (low_val < dc_low) and \
                           downtrend and \
                           volume_confirm and \
                           short_bias
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long: exit on Donchian low break or trend reversal
            signals[i] = 0.25
            if low_val < dc_low or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short: exit on Donchian high break or trend reversal
            signals[i] = -0.25
            if high_val > dc_high or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian_20_WeeklyPivot_Volume_1dTrend_v1"
timeframe = "6h"
leverage = 1.0