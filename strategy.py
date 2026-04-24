#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1w ADX (trend strength) - only trade when ADX > 25 to avoid choppy markets.
- Volume: Current 12h volume > 1.5 * 20-period 1w volume MA to confirm breakouts.
- Entry: Long when price breaks above 20-period Donchian high AND 1w ADX > 25 AND volume spike.
         Short when price breaks below 20-period Donchian low AND 1w ADX > 25 AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation or ADX < 20 (trend weakening).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Donchian channels work in both bull and bear markets by capturing breakouts, while ADX filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 12h (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for ADX and volume MA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1w ADX (14-period)
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    df_1w_volume = df_1w['volume'].values
    
    # True Range components
    tr1 = pd.Series(df_1w_high).diff().abs()
    tr2 = (pd.Series(df_1w_high) - pd.Series(df_1w_close).shift(1)).abs()
    tr3 = (pd.Series(df_1w_low) - pd.Series(df_1w_close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    dm_plus = pd.Series(df_1w_high).diff()
    dm_minus = -pd.Series(df_1w_low).diff()
    dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
    dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
    
    # Smoothed TR and DM
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume MA on 1w
    vol_ma_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Need enough bars for Donchian and 1w indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and trend filter
            if volume_spike[i] and adx_val > 25:
                # Bullish: price breaks above Donchian high
                if curr_high > period20_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian low
                elif curr_low < period20_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loss of volume confirmation OR trend weakening
            if curr_low < period20_low[i] or not volume_spike[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loss of volume confirmation OR trend weakening
            if curr_high > period20_high[i] or not volume_spike[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wADX_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0