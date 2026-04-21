#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index (CI) regime filter + 1d Donchian breakout + volume confirmation
# In trending regimes (CI < 38.2): trade breakouts in trend direction
# In ranging regimes (CI > 61.8): mean-revert at Donchian extremes
# Uses 1d Donchian channels for structure, 12h CI for regime, volume for confirmation
# Target: 12-37 trades/year by requiring regime alignment + breakout/pullback + volume confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian high/low
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Load 12h for Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Choppiness Index (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align indices
    
    # ATR(14)
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Absolute price change
    abs_change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    
    # Sum of absolute changes over 14 periods
    sum_abs_change = pd.Series(abs_change).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CI = 100 * log10(sum(abs_change)/atr) / log10(14)
    ci_12h = 100 * np.log10(sum_abs_change / atr_12h) / np.log10(14)
    
    # Align 12h CI to 12h (no additional delay needed for CI)
    ci_aligned = align_htf_to_ltf(prices, df_12h, ci_12h)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(ci_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Calculate 20-period volume average
        vol_lookback_start = max(0, i - 19)
        vol_window = prices['volume'].iloc[vol_lookback_start:i+1].values
        vol_ma_20 = np.mean(vol_window)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma_20
        
        # Regime filters
        trending = ci_aligned[i] < 38.2   # trending regime
        ranging = ci_aligned[i] > 61.8    # ranging regime
        
        if position == 0:
            # Enter long: breakout above Donchian high in trending OR pullback to support in ranging
            if (price > donchian_high_aligned[i] and volume_confirm and trending) or \
               (price <= donchian_low_aligned[i] * 1.005 and volume_confirm and ranging):  # near support
                signals[i] = 0.25
                position = 1
            # Enter short: breakout below Donchian low in trending OR pullback to resistance in ranging
            elif (price < donchian_low_aligned[i] and volume_confirm and trending) or \
                 (price >= donchian_high_aligned[i] * 0.995 and volume_confirm and ranging):  # near resistance
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: break below Donchian low in trending OR reach resistance in ranging
                if (price < donchian_low_aligned[i] and trending) or \
                   (price >= donchian_high_aligned[i] * 0.995 and ranging):
                    exit_signal = True
            elif position == -1:
                # Exit short: break above Donchian high in trending OR reach support in ranging
                if (price > donchian_high_aligned[i] and trending) or \
                   (price <= donchian_low_aligned[i] * 1.005 and ranging):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_CI_Donchian_Regime_Volume"
timeframe = "12h"
leverage = 1.0