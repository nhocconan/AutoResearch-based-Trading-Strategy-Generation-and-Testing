#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above Donchian upper band with volume > 1.5x 20-bar avg volume AND 1d ADX > 25.
Short when price breaks below Donchian lower band with volume > 1.5x 20-bar avg volume AND 1d ADX > 25.
Exit when price touches the opposite Donchian band or ADX < 20 (trend weakens).
Uses 4h for execution and volume, 1d for ADX trend filter.
Designed to capture strong trending moves with volume confirmation, works in both bull and bear markets.
Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    plus_dm = np.diff(high_1d, prepend=high_1d[0])
    minus_dm = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_period = 14
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=atr_period, min_periods=atr_period).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    adx_strong = adx > 25
    adx_weak = adx < 20
    
    # Align 1d ADX to 4h timeframe
    adx_strong_aligned = align_htf_to_ltf(prices, df_1d, adx_strong)
    adx_weak_aligned = align_htf_to_ltf(prices, df_1d, adx_weak)
    
    # Get 4h data for execution and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_period = 20
    upper_band = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate 4h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to primary timeframe (4h)
    upper_band_aligned = align_htf_to_ltf(prices, df_4h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_4h, lower_band)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(adx_strong_aligned[i]) or
            np.isnan(adx_weak_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper_band_aligned[i]
        breakout_lower = close[i] < lower_band_aligned[i]
        
        # Exit conditions
        exit_long = close[i] < lower_band_aligned[i] or adx_weak_aligned[i]
        exit_short = close[i] > upper_band_aligned[i] or adx_weak_aligned[i]
        
        if position == 0:
            # Long: break above upper band with volume confirmation and strong ADX
            if breakout_upper and volume_confirmed and adx_strong_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume confirmation and strong ADX
            elif breakout_lower and volume_confirmed and adx_strong_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch lower band or ADX weakens
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch upper band or ADX weakens
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ADX_Trend"
timeframe = "4h"
leverage = 1.0