#!/usr/bin/env python3
# 6h_1w_1d_adx_slope_volume_v1
# Strategy: ADX slope on weekly timeframe as trend filter, combined with 1d ADX strength and volume confirmation on 6h
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Weekly ADX slope identifies trend acceleration/deceleration. 
# In strong uptrend (rising weekly ADX), buy pullbacks when 1d ADX confirms strength and volume spikes.
# In strong downtrend (rising weekly ADX), sell bounces under same conditions.
# Works in bull by catching momentum continuations, in bear by catching trend continuations.
# Uses ADX slope to avoid whipsaws in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_adx_slope_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly ADX (14-period) for trend acceleration
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])],
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_ma / (tr_ma + 1e-10)
    di_minus = 100 * dm_minus_ma / (tr_ma + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ADX slope (3-period change) - rising ADX = strengthening trend
    adx_slope_1w = np.diff(adx_1w, n=3, prepend=adx_1w[0])  # 3-period difference
    
    # Daily ADX (14-period) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_d = high_1d[1:] - low_1d[1:]
    tr2_d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])],
                           np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    
    # Directional Movement
    dm_plus_d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                         np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus_d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                          np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_d = np.concatenate([[0], dm_plus_d])
    dm_minus_d = np.concatenate([[0], dm_minus_d])
    
    # Smoothed values
    tr_ma_d = pd.Series(tr_d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_ma_d = pd.Series(dm_plus_d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_ma_d = pd.Series(dm_minus_d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus_d = 100 * dm_plus_ma_d / (tr_ma_d + 1e-10)
    di_minus_d = 100 * dm_minus_ma_d / (tr_ma_d + 1e-10)
    dx_d = 100 * np.abs(di_plus_d - di_minus_d) / (di_plus_d + di_minus_d + 1e-10)
    adx_1d = pd.Series(dx_d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_slope_1w)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation (20-period average) on 6h
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_avg)  # Strong volume spike
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Warmup period
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_slope_1w_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions:
        # Long: Rising weekly ADX (trend strengthening) AND strong daily ADX AND volume spike
        # Short: Rising weekly ADX (trend strengthening) AND strong daily ADX AND volume spike
        # Direction determined by price relative to short-term momentum
        
        # Short-term momentum (5-period price change)
        mom_5 = (close[i] - close[i-5]) / close[i-5] if i >= 5 else 0
        
        long_condition = (adx_slope_1w_aligned[i] > 0) and (adx_1d_aligned[i] > 25) and vol_spike[i] and (mom_5 > -0.01)
        short_condition = (adx_slope_1w_aligned[i] > 0) and (adx_1d_aligned[i] > 25) and vol_spike[i] and (mom_5 < 0.01)
        
        # Exit when trend weakens (ADX slope turns negative) or ADX weakens
        exit_long = position == 1 and ((adx_slope_1w_aligned[i] <= 0) or (adx_1d_aligned[i] < 20))
        exit_short = position == -1 and ((adx_slope_1w_aligned[i] <= 0) or (adx_1d_aligned[i] < 20))
        
        # Trading logic
        if long_condition and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_condition and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals