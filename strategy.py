#!/usr/bin/env python3
# 4h_1d_cci_rsi_volume_v1
# Strategy: 4h CCI mean reversion with RSI filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions. In ranging markets, mean reversion works well.
# We use 1d ADX to detect ranging markets (ADX < 25) and only trade mean reversion in those conditions.
# Volume confirmation ensures genuine interest. Low frequency (~20-40/year) to minimize fee drag.
# Works in both bull and bear markets by focusing on ranging conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX for ranging market detection (ADX < 25 = ranging)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values
    atr[tr_period] = np.nansum(tr[1:tr_period+1])
    dm_plus_smooth[tr_period] = np.nansum(dm_plus[1:tr_period+1])
    dm_minus_smooth[tr_period] = np.nansum(dm_minus[1:tr_period+1])
    
    # Wilder's smoothing
    for i in range(tr_period + 1, len(tr)):
        atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # Avoid division by zero
    dir_plus = np.zeros_like(atr)
    dir_minus = np.zeros_like(atr)
    valid_atr = atr != 0
    dir_plus[valid_atr] = 100 * dm_plus_smooth[valid_atr] / atr[valid_atr]
    dir_minus[valid_atr] = 100 * dm_minus_smooth[valid_atr] / atr[valid_atr]
    
    dx = np.zeros_like(dir_plus)
    dir_sum = dir_plus + dir_minus
    valid_dir = dir_sum != 0
    dx[valid_dir] = 100 * np.abs(dir_plus[valid_dir] - dir_minus[valid_dir]) / dir_sum[valid_dir]
    
    adx = np.zeros_like(dx)
    adx[2*tr_period:] = np.nanmean(dx[tr_period:2*tr_period])  # Initial value
    for i in range(2*tr_period + 1, len(dx)):
        adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # CCI calculation (20-period)
    tp = (high + low + close) / 3  # Typical Price
    cci_period = 20
    cci_sma = pd.Series(tp).rolling(window=cci_period, min_periods=cci_period).mean()
    cci_mad = pd.Series(tp).rolling(window=cci_period, min_periods=cci_period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (tp - cci_sma) / (0.015 * cci_mad)
    cci_values = cci.values
    
    # RSI filter (14-period)
    rsi_period = 14
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(cci_period, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(cci_values[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Range filter: only trade in ranging markets (ADX < 25)
        ranging = adx_1d_aligned[i] < 25
        
        # Entry logic: CCI mean reversion + RSI filter + volume + ranging
        if (cci_values[i] < -100 and rsi_values[i] < 30 and  # Oversold
            vol_confirm[i] and ranging and position != 1):
            position = 1
            signals[i] = 0.25
        elif (cci_values[i] > 100 and rsi_values[i] > 70 and  # Overbought
              vol_confirm[i] and ranging and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: CCI returns to normal range or market starts trending
        elif position == 1 and (cci_values[i] > -50 or not ranging):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci_values[i] < 50 or not ranging):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals