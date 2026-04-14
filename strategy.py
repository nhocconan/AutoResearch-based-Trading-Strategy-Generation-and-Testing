#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and 1d ADX trend filter
# Donchian(20) breakout captures momentum with clear entry/exit levels
# Volume > 1.5x average confirms institutional participation
# ADX > 25 ensures we only trade in trending markets, avoiding chop
# Works in bull/bear as ADX filter adapts to trend strength
# Target: 20-40 trades/year per symbol (80-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    
    # Load 1d data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Volume confirmation: 12h average volume
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=10, min_periods=10).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # ADX calculation on 1d data (14 periods)
    # True Range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channel (20 periods) on 4h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, dc_len, 14)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 12h average
        volume_confirmed = volume[i] > 1.5 * vol_ma_12h_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Enter long: Donchian breakout above + volume + trend
            if (close[i] > dc_upper[i] and 
                volume_confirmed and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + volume + trend
            elif (close[i] < dc_lower[i] and 
                  volume_confirmed and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Donchian breakdown or loss of trend
            if close[i] < dc_lower[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Donchian breakout or loss of trend
            if close[i] > dc_upper[i] or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_1d_Donchian_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0