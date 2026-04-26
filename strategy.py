#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_RegimeFilter_v1
Hypothesis: Trade daily breakouts from Camarilla R1/S1 levels with weekly EMA50 trend filter, volume confirmation, and choppiness regime filter. Only trade in trending weekly markets (ADX > 25) to avoid whipsaws in ranging conditions. Uses discrete position size 0.25 to minimize fee drag. Designed to work in both bull and bear markets by following the weekly trend direction.
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
    
    # Get weekly data for trend filter and choppiness regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 from previous daily bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla R1 and S1 (tighter levels for daily breakouts)
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 12
    
    # Weekly EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly ADX for regime filter (trending vs ranging)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - np.roll(close_1w, 1)[1:])
    tr3 = np.abs(low_1w[1:] - np.roll(close_1w, 1)[1:])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.maximum(atr, 1e-10)
    di_minus = 100 * dm_minus_smooth / np.maximum(atr, 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-10)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align all weekly data to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=0)
    
    # Align daily Camarilla levels to daily timeframe (no delay needed as we use previous bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), ADX (14+14=28), volume MA (20)
    start_idx = max(50, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend alignment
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending_regime = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + weekly uptrend + trending regime
            long_breakout = close[i] > camarilla_r1_aligned[i]
            long_signal = long_breakout and volume_spike[i] and weekly_uptrend and trending_regime
            
            # Short: price breaks below S1 + volume spike + weekly downtrend + trending regime
            short_breakout = close[i] < camarilla_s1_aligned[i]
            short_signal = short_breakout and volume_spike[i] and weekly_downtrend and trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches S1 level OR weekly trend turns down OR regime becomes ranging
            if (close[i] < camarilla_s1_aligned[i] or not weekly_uptrend or not trending_regime):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R1 level OR weekly trend turns up OR regime becomes ranging
            if (close[i] > camarilla_r1_aligned[i] or not weekly_downtrend or not trending_regime):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_RegimeFilter_v1"
timeframe = "1d"
leverage = 1.0