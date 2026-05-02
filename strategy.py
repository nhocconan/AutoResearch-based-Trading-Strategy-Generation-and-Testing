#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction (from 1w Camarilla H4/L4), volume confirmation (>1.5x average), and ADX trend filter (ADX > 25)
# Weekly Camarilla H4/L4 provides institutional reference levels from prior week.
# Donchian breakout captures momentum; volume confirms institutional participation.
# ADX filter ensures trades only in trending markets, avoiding whipsaws in ranges.
# Discrete sizing 0.25 to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year).
# Primary timeframe: 6h, HTF: 1w for Camarilla levels, 1d for ADX.

name = "6h_Donchian20_Breakout_1wCamarilla_H4L4_Volume_ADX"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla H4 and L4 from 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior week's high, low, close for Camarilla calculation
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    prev_close_1w = df_1w['close'].shift(1).values
    
    # Camarilla H4 and L4 levels (more significant than H3/L3)
    camarilla_h4_1w = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 2
    camarilla_l4_1w = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 2
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_h4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    camarilla_l4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    
    # Daily ADX for trend filter (from 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_period = 14
    tr_smoothed = wilders_smoothing(tr, atr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / np.where(tr_smoothed == 0, 1e-10, tr_smoothed)
    di_minus = 100 * dm_minus_smoothed / np.where(tr_smoothed == 0, 1e-10, tr_smoothed)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1e-10, (di_plus + di_minus))
    adx = wilders_smoothing(dx, atr_period)
    
    # Align daily ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian(20) channels on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 1.5x 20-period average (balanced to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(lookback, 30) + atr_period
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h4_1w_aligned[i]) or np.isnan(camarilla_l4_1w_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high AND price > weekly H4 AND volume spike AND ADX > 25
            if (close[i] > highest_high[i] and 
                close[i] > camarilla_h4_1w_aligned[i] and 
                volume_spike[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND price < weekly L4 AND volume spike AND ADX > 25
            elif (close[i] < lowest_low[i] and 
                  close[i] < camarilla_l4_1w_aligned[i] and 
                  volume_spike[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian low OR weekly L4 OR ADX < 20 (trend weakening)
            if (close[i] < lowest_low[i] or 
                close[i] < camarilla_l4_1w_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian high OR weekly H4 OR ADX < 20 (trend weakening)
            if (close[i] > highest_high[i] or 
                close[i] > camarilla_h4_1w_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals