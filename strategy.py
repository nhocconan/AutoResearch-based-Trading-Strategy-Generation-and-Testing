#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX trend strength + 1d Williams %R overbought/oversold + volume confirmation
# ADX > 25 indicates strong trend for trend-following entries
# Williams %R on 1d: > -20 overbought (short), < -80 oversold (long)
# Volume filter: current 4h volume > 1.5x 20-period average to avoid low-volume false signals
# Designed to capture strong trends with momentum confirmation while avoiding choppy markets
# Target: 20-35 trades/year to avoid fee drag
name = "4h_ADX_WilliamsR_1dVolume_v1"
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
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 4h ADX (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    # Pad arrays to match length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 4h Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: ADX > 25 (strong trend) + Williams %R < -80 (oversold) + volume
            if adx[i] > 25 and williams_r_aligned[i] < -80 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: ADX > 25 (strong trend) + Williams %R > -20 (overbought) + volume
            elif adx[i] > 25 and williams_r_aligned[i] > -20 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: ADX < 20 (weakening trend) OR Williams %R > -50 (overbought)
            if adx[i] < 20 or williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: ADX < 20 (weakening trend) OR Williams %R < -50 (oversold)
            if adx[i] < 20 or williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals