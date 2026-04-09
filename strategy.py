#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_regime_v1
# Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and ADX regime filter.
# Long: Price breaks above 20-period 1w Donchian upper band with volume > 1.5x 20d average and ADX > 25 (trending)
# Short: Price breaks below 20-period 1w Donchian lower band with volume > 1.5x 20d average and ADX > 25 (trending)
# Exit: Price returns to 20-period 1w Donchian midpoint or opposite break
# Uses 1d primary timeframe with 1w HTF for Donchian calculation and ADX.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in both bull and bear markets by capturing strong trending moves with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Donchian channels and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) for 1w
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 1w Donchian channels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate ADX (14-period) for 1w
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with indices
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_period = 14
    atr_1w = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1w
    di_minus = 100 * dm_minus_smooth / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) != 0, dx, 0)
    adx_1w = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Align 1w ADX to 1d timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: ADX > 25 indicates trending market
        regime_filter = adx_1w_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: Price returns to midpoint or breaks below lower band (opposite signal)
            if close[i] <= donchian_mid_aligned[i] or close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to midpoint or breaks above upper band (opposite signal)
            if close[i] >= donchian_mid_aligned[i] or close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above upper band with volume confirmation and trending regime
            if close[i] > donchian_high_aligned[i] and volume_confirmed and regime_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower band with volume confirmation and trending regime
            elif close[i] < donchian_low_aligned[i] and volume_confirmed and regime_filter:
                position = -1
                signals[i] = -0.25
    
    return signals