#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Triple EMA crossover with 1d volume confirmation and 1w ADX trend filter
# - Triple EMA (8,21,55) for trend changes: long when EMA8>EMA21>EMA55, short when EMA8<EMA21<EMA55
# - 1d volume > 1.5x 20-period average for conviction
# - 1w ADX(14) > 25 for strong trend (avoid ranging markets)
# - Exit when EMA8 crosses back below/above EMA21
# - Designed to capture strong trends while avoiding choppy markets
# - Target: 25-40 trades/year to minimize fee drag

name = "4h_TripleEMA_1dVolume_1wADX_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w data
    # True Range
    tr1 = np.abs(df_1w['high'] - df_1w['low'])
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = df_1w['high'] - df_1w['high'].shift(1)
    down_move = df_1w['low'].shift(1) - df_1w['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Triple EMA (8,21,55) on 4h data
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(ema55[i]) or \
           np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 6.0)
        
        # Trend filter: 1w ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Look for long entry: bullish EMA alignment + volume + trend
            if ema8[i] > ema21[i] > ema55[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: bearish EMA alignment + volume + trend
            elif ema8[i] < ema21[i] < ema55[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when EMA8 crosses below EMA21
            if ema8[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when EMA8 crosses above EMA21
            if ema8[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals