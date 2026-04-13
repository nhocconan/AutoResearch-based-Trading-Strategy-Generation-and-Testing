#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d ATR-scaled volume filter and 1w chop regime.
    # Long when price breaks above Donchian(20) high + volume/ATR > 1.5x 20-period average + CHOP_1w > 61.8 (range).
    # Short when price breaks below Donchian(20) low + volume/ATR > 1.5x 20-period average + CHOP_1w > 61.8 (range).
    # Exit when price crosses Donchian(20) midpoint.
    # Uses ATR-scaled volume filter to adapt to volatility and chop regime to avoid trend-following false breakouts.
    # Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Get 1d data for ATR-scaled volume filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate True Range (TR) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR(14) on 1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-scaled volume average (20-period) on 1d: volume / ATR
    vol_atr_ratio_1d = volume_1d / np.maximum(atr_1d, 1e-10)
    vol_atr_ma_1d = pd.Series(vol_atr_ratio_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF ATR-scaled volume MA to 12h timeframe
    vol_atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_atr_ma_1d)
    
    # Get 1w data for chop regime (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range (TR) on 1w
    tr1_1w = np.abs(high_1w - low_1w)
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    tr_1w[0] = tr1_1w[0]  # First period
    
    # Calculate ATR(14) on 1w
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods on 1w
    hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Calculate Chopiness Index (CHOP) on 1w
    chop_denom = atr_1w * 14
    chop_num = hh_1w - ll_1w
    chop_1w = np.where(chop_denom != 0, 100 * np.log10(chop_num / chop_denom) / np.log10(14), 50)
    
    # Align HTF chop to 12h timeframe
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(vol_atr_ma_1d_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume/ATR ratio > 1.5x 20-period average
        vol_atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_atr_ratio_1d)
        volume_confirm = vol_atr_ratio_1d_aligned[i] > 1.5 * vol_atr_ma_1d_aligned[i]
        
        # Regime filter: CHOP_1w > 61.8 indicates ranging market (good for breakout fade)
        regime_filter = chop_1w_aligned[i] > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Entry conditions: breakout + volume + regime
        long_signal = long_breakout and volume_confirm and regime_filter
        short_signal = short_breakout and volume_confirm and regime_filter
        
        # Exit conditions: price crosses Donchian midpoint
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_donchian_vol_atr_chop_v1"
timeframe = "12h"
leverage = 1.0