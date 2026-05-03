#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
# Uses 1w ADX(14) > 25 for strong trend regime (both bull and bear).
# Entry: price breaks above Donchian(20) high with volume > 1.8x 20-period MA for longs,
#        or breaks below Donchian(20) low with volume spike for shorts.
# Exit: ATR(14) trailing stop (2.5x ATR) or Donchian(10) opposite breakout.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Donchian provides clear breakout levels; 1w ADX filters weak/choppy markets;
# volume confirmation reduces false breakouts. Works in bull via trend-following breakouts
# and in bear via short breakdowns with strong trend alignment.

name = "6h_Donchian20_1wADX_Volume_ATR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1w = np.concatenate([[tr_1w[0]], tr_1w])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = WilderSmoothing(tr_1w, 14)
    dm_plus_smooth = WilderSmoothing(dm_plus, 14)
    dm_minus_smooth = WilderSmoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, 100 * dm_plus_smooth / atr_1w, 0)
    di_minus = np.where(atr_1w != 0, 100 * dm_minus_smooth / atr_1w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = WilderSmoothing(dx, 14)
    
    # Align 1w ADX to 6h timeframe (wait for completed 1w bar)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Donchian(20) channels on 6h
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate ATR(14) for stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[tr[0]], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume regime: current 6h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long ATR stop
    lowest_since_entry = 0.0   # for short ATR stop
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_1w_aligned[i]) or np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or np.isnan(donchian_high_10[i]) or 
            np.isnan(donchian_low_10[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        adx_val = adx_1w_aligned[i]
        dh20 = donchian_high_20[i]
        dl20 = donchian_low_20[i]
        dh10 = donchian_high_10[i]
        dl10 = donchian_low_10[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Determine strong trend regime (ADX > 25)
        is_strong_trend = adx_val > 25
        
        # Update highest/lowest since entry for ATR stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i]) if lowest_since_entry != 0 else low[i]
        
        # Entry logic
        if position == 0:
            # Long: break above Donchian(20) high with volume spike in strong trend
            if close_val > dh20 and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = high[i]
            # Short: break below Donchian(20) low with volume spike in strong trend
            elif close_val < dl20 and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = low[i]
        elif position == 1:
            # Long exit: ATR stoploss OR Donchian(10) low breakout
            atr_stop = highest_since_entry - (2.5 * atr_val)
            if close_val < atr_stop or close_val < dl10:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ATR stoploss OR Donchian(10) high breakout
            atr_stop = lowest_since_entry + (2.5 * atr_val)
            if close_val > atr_stop or close_val > dh10:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals