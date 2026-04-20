#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Chande Kroll Stop with Volume Confirmation and ADX Trend Filter
# Chande Kroll Stop adapts to volatility, providing dynamic support/resistance.
# Works in both bull and bear markets by using ATR-based bands that tighten in low vol and widen in high vol.
# Volume confirmation ensures breakouts have conviction.
# ADX filter ensures we only trade in trending conditions, avoiding whipsaws in ranges.
# Target: 20-40 trades/year on 4h timeframe to minimize fee drag.

name = "4h_ChandeKrollStop_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for ADX (trend filter) - calculated once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    tr_period = len(tr)
    if tr_period < period:
        return np.zeros(n)
        
    atr = smooth_wilder(tr, period)
    dm_plus_smooth = smooth_wilder(dm_plus, period)
    dm_minus_smooth = smooth_wilder(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, period)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Chande Kroll Stop calculation on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR for Chande Kroll (using 10-period ATR)
    tr_4h1 = np.abs(high[1:] - low[1:])
    tr_4h2 = np.abs(high[1:] - close[:-1])
    tr_4h3 = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.max([tr_4h1[0], tr_4h2[0], tr_4h3[0]])], np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))])
    
    atr_4h = np.zeros_like(close)
    atr_period = 10
    if len(tr_4h) >= atr_period:
        atr_4h[atr_period-1] = np.nansum(tr_4h[:atr_period])
        for i in range(atr_period, len(tr_4h)):
            atr_4h[i] = atr_4h[i-1] - (atr_4h[i-1] / atr_period) + tr_4h[i]
    
    # Chande Kroll Stop: ± multiple of ATR from recent high/low
    # For long: stop below recent low; for short: stop above recent high
    # We use the stops as dynamic support/resistance for entry
    mult = 2.0
    chandek_long_stop = np.zeros_like(close)
    chandek_short_stop = np.zeros_like(close)
    
    # Calculate rolling max/min for stops
    for i in range(len(close)):
        if i < atr_period:
            chandek_long_stop[i] = np.nan
            chandek_short_stop[i] = np.nan
        else:
            # Long stop: recent low - ATR * mult
            recent_low = np.min(low[max(0, i-atr_period+1):i+1])
            chandek_long_stop[i] = recent_low - (atr_4h[i] * mult)
            # Short stop: recent high + ATR * mult
            recent_high = np.max(high[max(0, i-atr_period+1):i+1])
            chandek_short_stop[i] = recent_high + (atr_4h[i] * mult)
    
    # Volume filter
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        # Get values
        close_val = close[i]
        ch_long_stop = chandek_long_stop[i]
        ch_short_stop = chandek_short_stop[i]
        vol_ratio_val = vol_ratio[i]
        adx_val = adx_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ch_long_stop) or np.isnan(ch_short_stop) or 
            np.isnan(vol_ratio_val) or np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ADX threshold for trending market
        if adx_val < 25:
            # Not trending enough, flatten if in position
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above Chande Kroll long stop (support) with volume confirmation
            if close_val > ch_long_stop and vol_ratio_val > 1.8:
                signals[i] = 0.25
                position = 1
            # Short: Price below Chande Kroll short stop (resistance) with volume confirmation
            elif close_val < ch_short_stop and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: Hold while price above long stop, exit if breaks below
            if close_val < ch_long_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: Hold while price below short stop, exit if breaks above
            if close_val > ch_short_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals