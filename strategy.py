#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h ADX trend filter and volume confirmation.
# Williams %R < -80 = oversold (long), > -20 = overbought (short).
# ADX > 25 confirms trend strength to avoid whipsaws in ranging markets.
# Volume > 1.5x 20-period average ensures participation.
# Uses 4h for trend direction (ADX) and 1h for entry timing (Williams %R).
# Target: 15-30 trades/year to avoid fee drag. Works in bull/bear via trend filter.
name = "1h_WilliamsR_4hADX_VolumeFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # ADX (14-period) from 4h for trend strength
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 4h ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold), ADX > 25 (trending), volume filter
            long_cond = (williams_r[i] < -80) and (adx_aligned[i] > 25) and volume_filter[i]
            # Short conditions: Williams %R > -20 (overbought), ADX > 25 (trending), volume filter
            short_cond = (williams_r[i] > -20) and (adx_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Williams %R > -50 (exit oversold) OR ADX < 20 (weak trend)
            if williams_r[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Williams %R < -50 (exit overbought) OR ADX < 20 (weak trend)
            if williams_r[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals