#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TrixVolumeSpike_ChoppyRegime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate TRIX on daily close
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = np.diff(ema3, prepend=ema3[0]) / ema3
    trix = pd.Series(trix_raw).ewm(span=12, adjust=False, min_periods=12).mean().values * 100
    
    # Align TRIX to 4h
    trix_4h = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate Choppy Index on daily high/low/close (14-period)
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_close14 = pd.Series(df_1d['close'].values).rolling(window=14, min_periods=14).max().values
    lowest_close14 = pd.Series(df_1d['close'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr14 / (highest_close14 - lowest_close14)) / np.log10(14)
    chop = np.where((highest_close14 - lowest_close14) == 0, 50, chop)
    
    # Align Chop to 4h
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: above 2x 24-period average (24*4h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_4h[i]) or np.isnan(chop_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume spike confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        # Chop regime: chop > 61.8 = range (mean revert), chop < 38.2 = trending (trend follow)
        chop_value = chop_4h[i]
        in_range = chop_value > 61.8
        in_trend = chop_value < 38.2
        
        if position == 0:
            # Long: TRIX crossing above zero in ranging market (mean reversion long)
            if (trix_4h[i] > 0 and trix_4h[i-1] <= 0 and 
                in_range and vol_ok and in_session):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crossing below zero in ranging market (mean reversion short)
            elif (trix_4h[i] < 0 and trix_4h[i-1] >= 0 and 
                  in_range and vol_ok and in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero or chop shifts to trending
            if (trix_4h[i] < 0 and trix_4h[i-1] >= 0) or not in_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero or chop shifts to trending
            if (trix_4h[i] > 0 and trix_4h[i-1] <= 0) or not in_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals