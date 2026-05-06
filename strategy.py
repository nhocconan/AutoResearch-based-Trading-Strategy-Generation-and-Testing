#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h ADX trend filter and volume spike
# Williams %R(14) > -20 = overbought (short), < -80 = oversold (long)
# 12h ADX > 25 confirms trend strength to avoid whipsaw in ranging markets
# Volume spike (>2.0x 20-bar average) confirms momentum behind moves
# Designed for 6b timeframe to capture swings in both bull/bear markets
# Target: 60-120 total trades over 4 years (15-30/year) with discrete sizing 0.25

name = "6h_WilliamsR_12hADX_VolumeSpike_v1"
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
    
    # Calculate HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14) trend filter
    # TR calculation
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = np.concatenate([[np.nan], high_12h[1:] - high_12h[:-1]])
    down_move = np.concatenate([[np.nan], low_12h[:-1] - low_12h[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    plus_di = 100 * plus_dm_smooth / atr_12h
    minus_di = 100 * minus_dm_smooth / atr_12h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND volume spike
            if williams_r[i] < -80 and adx_12h_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND volume spike
            elif williams_r[i] > -20 and adx_12h_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -50 (return from oversold) OR ADX < 20 (trend weak)
            if williams_r[i] > -50 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -50 (return from overbought) OR ADX < 20 (trend weak)
            if williams_r[i] < -50 or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals