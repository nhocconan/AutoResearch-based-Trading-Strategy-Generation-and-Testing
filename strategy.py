#!/usr/bin/env python3
"""
12h_1w_Donchian_Breakout_Volume_v1
Hypothesis: Use weekly Donchian channels (20-period) with volume confirmation and ADX trend filter.
Long when price breaks above upper Donchian with volume > 1.5x average and ADX > 25.
Short when price breaks below lower Donchian with volume > 1.5x average and ADX > 25.
Only trade in direction of weekly EMA50 trend to avoid counter-trend whipsaws.
Targets 15-30 trades/year to minimize fee drag. Works in bull (follow trend breakouts) and bear (fade reversals at Donchian bands).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Donchian_Breakout_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === WEEKLY DONCHIAN CHANNEL (20-period) ===
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Upper band: highest high of last 20 weeks
    upper_donchian = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 weeks
    lower_donchian = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    upper_donchian_12h = align_htf_to_ltf(prices, df_1w, upper_donchian)
    lower_donchian_12h = align_htf_to_ltf(prices, df_1w, lower_donchian)
    
    # === WEEKLY EMA50 TREND FILTER ===
    ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1w, ema50)
    
    # === WEEKLY ADX (14-period) FOR TREND STRENGTH ===
    # Calculate True Range
    tr1 = weekly_high[1:] - weekly_low[1:]
    tr2 = np.abs(weekly_high[1:] - weekly_close[:-1])
    tr3 = np.abs(weekly_low[1:] - weekly_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((weekly_high[1:] - weekly_high[:-1]) > (weekly_low[:-1] - weekly_low[1:]), 
                       np.maximum(weekly_high[1:] - weekly_high[:-1], 0), 0)
    dm_minus = np.where((weekly_low[:-1] - weekly_low[1:]) > (weekly_high[1:] - weekly_high[:-1]), 
                        np.maximum(weekly_low[:-1] - weekly_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    atr = np.full(len(tr), np.nan)
    dm_plus_smooth = np.full(len(dm_plus), np.nan)
    dm_minus_smooth = np.full(len(dm_minus), np.nan)
    
    # Initial values
    if len(tr) >= 14:
        atr[13] = np.nansum(tr[1:15])  # First 14-period ATR
        dm_plus_smooth[13] = np.nansum(dm_plus[1:15])
        dm_minus_smooth[13] = np.nansum(dm_minus[1:15])
        
        # Wilder's smoothing for remaining values
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # Calculate DI+ and DI-
    di_plus = np.full(len(atr), np.nan)
    di_minus = np.full(len(atr), np.nan)
    dx = np.full(len(atr), np.nan)
    
    for i in range(14, len(atr)):
        if not np.isnan(atr[i]) and atr[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
            if (di_plus[i] + di_minus[i]) != 0:
                dx[i] = (np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])) * 100
    
    # Calculate ADX (smoothed DX)
    adx = np.full(len(dx), np.nan)
    if len(dx) >= 28:  # Need 14 for initial DX + 14 for smoothing
        adx[27] = np.nanmean(dx[14:28])  # First 14-period ADX
        for i in range(28, len(dx)):
            if not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_12h = align_htf_to_ltf(prices, df_1w, adx)
    
    # === VOLUME FILTER (20-period average) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(upper_donchian_12h[i]) or np.isnan(lower_donchian_12h[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(adx_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        # Trend strength filter: ADX > 25 indicates strong trend
        strong_trend = adx_12h[i] > 25
        
        # Trend filter: price above/below EMA50
        trend_up = close[i] > ema50_12h[i]
        
        # Breakout conditions
        breakout_up = high[i] > upper_donchian_12h[i] and vol_confirm and strong_trend
        breakout_down = low[i] < lower_donchian_12h[i] and vol_confirm and strong_trend
        
        # Entry logic: only trade in direction of weekly trend
        long_entry = breakout_up and trend_up
        short_entry = breakout_down and not trend_up
        
        # Exit logic: reverse signal or price returns to EMA50 (trend filter)
        long_exit = not breakout_up or close[i] < ema50_12h[i]
        short_exit = not breakout_down or close[i] > ema50_12h[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals