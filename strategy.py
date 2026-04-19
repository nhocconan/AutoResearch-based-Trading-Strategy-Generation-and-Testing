#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1d ADX trend filter
# Long when price breaks above 20-period 12h high + 1d volume > 2x average + 1d ADX > 25
# Short when price breaks below 20-period 12h low + 1d volume > 2x average + 1d ADX > 25
# Exit when price touches opposite Donchian band or ADX < 20 (trend weakens)
# Designed for 12h timeframe to reduce trade frequency and avoid fee drag
# Target: 15-25 trades/year to stay within 50-100 total over 4 years
name = "12h_Donchian_Volume_ADX_1d_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.concatenate([[0], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[0], low_1d[:-1] - low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_smooth / (atr_1d + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2x average
        vol_filter = volume[i] > 2 * vol_ma_1d_aligned[i]
        
        # ADX filter: trend strength
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume + strong trend
            if close[i] > donch_high[i] and vol_filter and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume + strong trend
            elif close[i] < donch_low[i] and vol_filter and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches Donchian low OR trend weakens
            if close[i] < donch_low[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches Donchian high OR trend weakens
            if close[i] > donch_high[i] or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals