#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w ATR regime filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for ATR-based regime detection (trending vs choppy) and volume spike filter.
- Donchian(20): Upper/lower bands from 20-period high/low on 1d timeframe.
- Regime: ATR(10)/ATR(30) ratio > 1.2 = trending (favor breakouts), < 0.8 = choppy (avoid breakouts).
- Entry: Long when price > Upper Band AND trending regime AND volume > 2.0 * 20-period average volume.
         Short when price < Lower Band AND trending regime AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Donchian breakout (price < Upper Band for long exit, price > Lower Band for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading breakouts in trending regimes, avoiding whipsaws in chop.
- Uses 1w HTF to avoid look-ahead and ensure proper alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w ATR(10) and ATR(30) for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need sufficient data for ATR30
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(10) and ATR(30)
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR ratio for regime: >1.2 = trending, <0.8 = choppy
    atr_ratio = atr10 / atr30
    
    # Align ATR ratio to 1d timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Calculate 1w volume average for confirmation (20-period)
    if len(df_1w) < 20:
        return np.zeros(n)
    
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate 1d Donchian(20) bands
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 30)  # Need 20 for Donchian, 30 for ATR30
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]) or
            np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade breakouts in trending markets (ATR ratio > 1.2)
        trending_regime = atr_ratio_aligned[i] > 1.2
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1w_aligned[i] if not np.isnan(vol_ma_20_1w_aligned[i]) else False
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price < Upper Band
            if position == 1:
                if curr_close < upper_band[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > Lower Band
            elif position == -1:
                if curr_close > lower_band[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with regime and volume filters
        if position == 0:
            # Long: price > Upper Band AND trending regime AND volume confirmation
            long_condition = (curr_close > upper_band[i] and 
                            trending_regime and
                            volume_confirm)
            
            # Short: price < Lower Band AND trending regime AND volume confirmation
            short_condition = (curr_close < lower_band[i] and 
                             trending_regime and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wATRRegime_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0