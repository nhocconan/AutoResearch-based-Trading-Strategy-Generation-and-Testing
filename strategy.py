#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ADX trend strength.
- Donchian breakout: Long when price > upper channel (20-period high), Short when price < lower channel (20-period low).
- ADX > 25 indicates trending market (trade breakouts), ADX < 20 indicates ranging (fade breakouts).
- Entry: Long on bullish breakout in trending regime (ADX>25) or bearish breakout in ranging regime (ADX<20).
         Short on bearish breakout in trending regime (ADX>25) or bullish breakout in ranging regime (ADX<20).
- Volume confirmation: current volume > 1.3 * 20-period volume MA (on 4h).
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
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
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
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_channel = high_roll.values
    lower_channel = low_roll.values
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for ADX and 4h bars for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        price = close[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if adx_val > 25:  # Trending regime: trade breakouts
                    # Bullish breakout: price > upper channel
                    if price > upper:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price < lower channel
                    elif price < lower:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime (ADX < 20): fade breakouts
                    # Fade bullish breakout: price > upper channel -> short
                    if price > upper:
                        signals[i] = -0.25
                        position = -1
                    # Fade bearish breakout: price < lower channel -> long
                    elif price < lower:
                        signals[i] = 0.25
                        position = 1
        elif position == 1:
            # Long exit: price re-enters Donchian channel or ADX drops to ranging
            if price < upper and price > lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Donchian channel or ADX drops to ranging
            if price < upper and price > lower or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dADXRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0