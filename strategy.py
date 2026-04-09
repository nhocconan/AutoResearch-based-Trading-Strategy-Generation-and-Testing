#!/usr/bin/env python3
# 12h_donchian_breakout_volume_chop_v1
# Hypothesis: 12h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Donchian(20) breakout captures trend continuation; volume > 1.5x 20-period average confirms institutional participation.
# Choppiness Index (CHOP) > 61.8 indicates ranging market (avoid false breakouts), CHOP < 38.2 indicates trending market (favor breakouts).
# Only trade breakouts in trending regimes (CHOP < 38.2) with volume confirmation.
# Uses 1d HTF for choppiness calculation (more stable regime detection).
# Discrete sizing (±0.25) to minimize fee churn. Target: 12-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for choppiness regime filter (more stable)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with index
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM for 1d
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed +DM, -DM, ATR
    atr_1d_smooth = pd.Series(atr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di_1d = 100 * plus_dm_smooth / atr_1d_smooth
    minus_di_1d = 100 * minus_dm_smooth / atr_1d_smooth
    
    # DX and Choppiness Index (CHOP)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    chop_1d = 100 * np.log10(pd.Series(dx_1d).rolling(window=14, min_periods=14).sum().values / np.log10(14)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 12h Donchian breakout
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 12h volume confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_window, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop_1d_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower band OR regime becomes ranging
            if close[i] < lowest_low[i] or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper band OR regime becomes ranging
            if close[i] > highest_high[i] or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper band + volume + trending regime
            if close[i] > highest_high[i] and volume_confirmed and trending_regime:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower band + volume + trending regime
            elif close[i] < lowest_low[i] and volume_confirmed and trending_regime:
                position = -1
                signals[i] = -0.25
    
    return signals