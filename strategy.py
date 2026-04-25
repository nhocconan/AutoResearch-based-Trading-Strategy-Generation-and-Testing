#!/usr/bin/env python3
"""
6h Weekly Donchian Breakout + Daily ADX Trend + Volume Spike
Hypothesis: Weekly Donchian channels identify major support/resistance. Breakouts with daily ADX>25 (strong trend) and volume spike capture momentum moves that persist through 6h timeframe. Works in bull/bear via ADX trend filter - only takes breakouts in direction of daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels
    highest_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1w, highest_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    plus_dm = np.diff(df_1d['high'], prepend=df_1d['high'][0])
    minus_dm = np.diff(df_1d['low'], prepend=df_1d['low'][0]) * -1
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(df_1d['high']))
    tr2 = np.abs(np.diff(df_1d['low']))
    tr3 = np.abs(np.diff(df_1d['close']))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[tr[0]], tr]) if len(tr) > 0 else np.array([0])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Daily trend direction from EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for weekly Donchian and daily indicators
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_high = high[i]
        curr_low = low[i]
        adx_val = adx_aligned[i]
        ema_trend = ema_34_aligned[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above weekly Donchian high AND ADX>25 AND price above daily EMA34 (uptrend)
            long_condition = (curr_high > donch_high) and (adx_val > 25) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below weekly Donchian low AND ADX>25 AND price below daily EMA34 (downtrend)
            short_condition = (curr_low < donch_low) and (adx_val > 25) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below weekly Donchian low or trend weakens
            if curr_close < donch_low or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above weekly Donchian high or trend weakens
            if curr_close > donch_high or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian_Breakout_DailyADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0