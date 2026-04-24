#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for ADX trend regime.
- Donchian channel (20-period) from prior 1d: Long when price > upper band, Short when price < lower band.
- Regime filter: Only trade when 1d ADX(14) > 25 (trending market) to avoid whipsaws in ranging markets.
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Uses actual Donchian calculation: upper = max(high, lookback), lower = min(low, lookback).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) from prior 1d bar
    # Upper band = max(high, lookback), Lower band = min(low, lookback)
    lookback = 20
    donch_upper = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    donch_lower = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align to 12h: use prior 1d's levels (already completed bar)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower)
    
    # 1d ADX(14) for regime filter
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    period = 14
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # +DM and -DM
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM and -DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Align to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, period*2, 20)  # Donchian + ADX + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in trending regime (ADX > 25)
            if adx_aligned[i] > 25:
                if close[i] > donch_upper_aligned[i] and volume_spike[i]:
                    # Buy on Donchian upper breakout in trending market
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donch_lower_aligned[i] and volume_spike[i]:
                    # Sell on Donchian lower breakdown in trending market
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to midpoint of Donchian channel or opposite break
            midpoint = (donch_upper_aligned[i] + donch_lower_aligned[i]) / 2
            if not np.isnan(midpoint):
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midpoint of Donchian channel or opposite break
            midpoint = (donch_upper_aligned[i] + donch_lower_aligned[i]) / 2
            if not np.isnan(midpoint):
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dADX_Regime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0