#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX filter and volume confirmation
# Long when price breaks above Donchian(20) high in 1d uptrend (ADX > 25 and +DI > -DI)
# Short when price breaks below Donchian(20) low in 1d downtrend (ADX > 25 and -DI > +DI)
# Uses volume > 1.5x average for confirmation to avoid false breakouts
# Targets 100-200 total trades over 4 years (25-50/year) by requiring trend + breakout + volume

name = "4h_donchian_1d_adx_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d ADX for trend direction
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[1:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    di_plus = np.where(atr_1d > 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d > 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilder_smooth(dx, 14)
    
    # Align 1d indicators to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(di_plus_aligned[i]) or np.isnan(di_minus_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine 1d trend direction: up if +DI > -DI, down if -DI > +DI (only when ADX > 25)
        is_uptrend = adx_aligned[i] > 25 and di_plus_aligned[i] > di_minus_aligned[i]
        is_downtrend = adx_aligned[i] > 25 and di_minus_aligned[i] > di_plus_aligned[i]
        
        if position == 1:  # long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] < lowest_low[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] > highest_high[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume confirmation in trending market
            if is_uptrend and close[i] > highest_high[i] and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            elif is_downtrend and close[i] < lowest_low[i] and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
    
    return signals