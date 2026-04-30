#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX25 trend filter and volume confirmation.
# Uses Donchian channels from 4h for breakout signals, 1d ADX > 25 to ensure trending market,
# volume > 1.8x 20-bar average for confirmation, and ATR(14) trailing stop (2.0x) for risk management.
# Discrete position sizing at ±0.25 to limit fee drag. Target: 80-150 total trades over 4 years (20-38/year).
# Session filter (08:00-20:00 UTC) to avoid low-liquidity periods.

name = "4h_Donchian20_1dADX25_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    
    tr_14 = np.zeros_like(tr_1d)
    dm_plus_14 = np.zeros_like(dm_plus)
    dm_minus_14 = np.zeros_like(dm_minus)
    
    tr_14[0] = np.mean(tr_1d[:atr_period])
    dm_plus_14[0] = np.mean(dm_plus[:atr_period])
    dm_minus_14[0] = np.mean(dm_minus[:atr_period])
    
    for i in range(1, len(tr_1d)):
        tr_14[i] = tr_14[i-1] - (tr_14[i-1] / atr_period) + tr_1d[i]
        dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / atr_period) + dm_plus[i]
        dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / atr_period) + dm_minus[i]
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = np.zeros_like(tr_1d)
    mask = (di_plus + di_minus) > 0
    dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / (di_plus[mask] + di_minus[mask])
    
    adx_1d = np.zeros_like(tr_1d)
    adx_1d[:atr_period] = np.nan
    for i in range(atr_period, len(dx)):
        if i == atr_period:
            adx_1d[i] = np.mean(dx[:i+1])
        else:
            adx_1d[i] = (adx_1d[i-1] * (atr_period - 1) + dx[i]) / atr_period
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian(20) on 4h
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR(14) for volatility and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(30, donchian_period, atr_period)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(adx_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_adx = adx_1d_aligned[i]
        curr_highest = highest_high[i]
        curr_lowest = lowest_low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, ADX > 25, volume confirmation
            if (curr_close > curr_highest and 
                curr_adx > 25 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price breaks below Donchian low, ADX > 25, volume confirmation
            elif (curr_close < curr_lowest and 
                  curr_adx > 25 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 2.0*ATR from highest point
            if curr_close < highest_since_entry - (2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest point
            if curr_close > lowest_since_entry + (2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals