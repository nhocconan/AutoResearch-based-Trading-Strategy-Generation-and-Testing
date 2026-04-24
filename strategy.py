#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1h ADX regime filter.
- Primary timeframe: 4h for execution, HTF: 1d for Donchian channels, 1h for ADX trend strength.
- ADX > 25 indicates trending market (breakout strategy), ADX < 20 indicates ranging (avoid breakouts).
- Entry: Long when price breaks above Donchian(20) upper band AND ADX > 25 (bullish breakout in trend).
         Short when price breaks below Donchian(20) lower band AND ADX > 25 (bearish breakout in trend).
- Exit: Opposite Donchian breakout or ADX regime shift to ranging (ADX < 20).
- Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h) to avoid false breakouts.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Get 1h data for ADX
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1h
    # True Range
    tr1 = pd.Series(df_1h['high']).diff().abs()
    tr2 = (pd.Series(df_1h['high']) - pd.Series(df_1h['low'].shift())).abs()
    tr3 = (pd.Series(df_1h['low']) - pd.Series(df_1h['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1h['high']).diff()
    down_move = -pd.Series(df_1h['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1h, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1h bars for ADX and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if adx_val > 25:  # Trending regime: breakout strategy
                    # Bullish breakout: price breaks above upper band
                    if curr_high > upper:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price breaks below lower band
                    elif curr_low < lower:
                        signals[i] = -0.25
                        position = -1
                # In ranging regime (ADX < 20): avoid breakouts, wait for trend
        elif position == 1:
            # Long exit: price breaks below lower band OR ADX drops to ranging
            if curr_low < lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper band OR ADX drops to ranging
            if curr_high > upper or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dChannels_1hADXRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0