#!/usr/bin/env python3
"""
6h Weekly Donchian Breakout with 1d Volume Spike and ADX Trend Filter
Hypothesis: Weekly Donchian channels (20-bar) on 6h capture major trend breaks.
Breakouts above weekly H20 or below weekly L20 with volume confirmation (>1.8x 20-bar vol MA)
and 1d ADX > 25 trend filter capture strong momentum moves in both bull and bear markets.
Uses ATR-based trailing stop (2.5*ATR) for risk control. Targets 50-150 total trades over 4 years
to avoid fee drag. Weekly structure provides stronger filters than daily, reducing whipsaws.
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
    
    # Get 1w data for weekly Donchian channels (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian H20/L20 (20-period high/low)
    donch_h20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donch_l20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    donch_h20_aligned = align_htf_to_ltf(prices, df_1w, donch_h20)
    donch_l20_aligned = align_htf_to_ltf(prices, df_1w, donch_l20)
    
    # Get 1d data for ADX trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr1[i] = max(high_1d[i] - low_1d[i], 
                     abs(high_1d[i] - close_1d[i-1]), 
                     abs(low_1d[i] - close_1d[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    tr_smooth = wilders_smoothing(tr1, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di = np.full_like(tr_smooth, np.nan)
    minus_di = np.full_like(tr_smooth, np.nan)
    dx = np.full_like(tr_smooth, np.nan)
    
    for i in range(14, len(tr_smooth)):
        if tr_smooth[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / tr_smooth[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / tr_smooth[i]) * 100
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # ADX = Wilder's smoothing of DX
    adx = wilders_smoothing(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 1d data for volume confirmation (call ONCE before loop)
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) for stoploss (6h)
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for weekly Donchian, 1d ADX, 1d volume MA, ATR to propagate
    start_idx = max(20, 30, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_h20_aligned[i]) or 
            np.isnan(donch_l20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donch_h20 = donch_h20_aligned[i]
        donch_l20 = donch_l20_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_1d = vol_ma_20_1d_aligned[i]
        atr = atr_14[i]
        
        # Volume confirmation: current 6h volume > 1.8 * 20-day average volume (scaled)
        # Scale 1d volume to 6h: approx 4x (4 six-hour bars in a day)
        vol_ma_6h_equiv = vol_ma_1d * 0.25  # Convert daily MA to 6h equivalent
        volume_confirm = curr_volume > 1.8 * vol_ma_6h_equiv
        
        # ADX trend filter: strong trend when ADX > 25
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long breakout: close above weekly H20 with volume confirmation and strong trend
            long_breakout = (curr_close > donch_h20) and volume_confirm and strong_trend
            # Short breakdown: close below weekly L20 with volume confirmation and strong trend
            short_breakout = (curr_close < donch_l20) and volume_confirm and strong_trend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 2.5 * atr  # Initial stop
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 2.5 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 2.5*ATR
            atr_stop = max(atr_stop, curr_high - 2.5 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 2.5*ATR
            atr_stop = min(atr_stop, curr_low + 2.5 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_1dADX25_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0