#!/usr/bin/env python3
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
    
    # Get weekly data for HTF trend context
    weekly = get_htf_data(prices, '1w')
    weekly_close = weekly['close'].values
    
    # Calculate weekly EMA40 for trend filter
    ema_40 = pd.Series(weekly_close).ewm(span=40, adjust=False, min_periods=40).mean().values
    weekly_trend = align_htf_to_ltf(prices, weekly, ema_40)
    
    # Get daily data for Camarilla pivot levels
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = daily_close + (daily_high - daily_low) * 1.1 / 12.0
    s1 = daily_close - (daily_high - daily_low) * 1.1 / 12.0
    
    # Align pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, daily, r1)
    s1_4h = align_htf_to_ltf(prices, daily, s1)
    
    # Calculate daily volume for volume confirmation
    daily_volume = daily['volume'].values
    avg_volume_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = align_htf_to_ltf(prices, daily, avg_volume_20)
    
    # Calculate daily ADX for trend strength filter (regime filter)
    # Simplified ADX calculation using daily data
    daily_tr = np.maximum(daily_high - daily_low,
                          np.maximum(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])),
                                     np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))))
    atr_14 = pd.Series(daily_tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])
    down_move = np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM and -DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smooth / (atr_14 + 1e-10))
    minus_di = 100 * (minus_dm_smooth / (atr_14 + 1e-10))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_4h = align_htf_to_ltf(prices, daily, adx)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_trend[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(volume_ratio[i]) or np.isnan(adx_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * average daily volume
        # Note: volume_ratio contains the 20-day average volume aligned to 4h
        # We need to compare current 4h volume to this daily average
        # Approximate: current 4h volume > 0.5 * daily average volume (since 4h is 1/6 of day)
        if volume[i] < 0.5 * volume_ratio[i]:
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        if adx_4h[i] < 25:
            signals[i] = 0.0
            continue
        
        # Long when price touches S1 support in uptrend (weekly close > weekly EMA40)
        if (close[i] <= s1_4h[i] and 
            weekly_trend[i] < close[i]):  # Uptrend: price above weekly EMA40
            signals[i] = 0.25
        # Short when price touches R1 resistance in downtrend (weekly close < weekly EMA40)
        elif (close[i] >= r1_4h[i] and 
              weekly_trend[i] > close[i]):  # Downtrend: price below weekly EMA40
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_WeeklyEMA40_Camarilla_Pivot_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0