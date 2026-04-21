#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_Volume_ADX_Filter
Hypothesis: Daily Donchian channel breakouts with volume confirmation and ADX trend filter capture strong trends while avoiding whipsaws. Works in bull/bear markets by only taking breakouts in direction of weekly trend (price vs EMA50 weekly). Targets 10-25 trades/year by requiring strong trend (ADX>25) and volume spike (2x average). Uses 1d as primary timeframe with 1w HTF for trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM-
    tr_period = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    dm_plus_period = pd.Series(dm_plus).ewm(alpha=1/period, adjust=False).mean()
    dm_minus_period = pd.Series(dm_minus).ewm(alpha=1/period, adjust=False).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly ADX for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w_arr, 14)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(adx_14_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate daily Donchian channels from previous day's data
        # Extract daily OHLC series
        high_d = prices['high'].values
        low_d = prices['low'].values
        close_d = prices['close'].values
        
        # Calculate Donchian channels (20-period)
        upper, lower = calculate_donchian_channels(high_d, low_d, 20)
        
        # Use previous bar's Donchian levels (breakout of previous 20-day range)
        prev_upper = upper[i-1]
        prev_lower = lower[i-1]
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price > weekly EMA50 for long, price < weekly EMA50 for short
        trend_long = price > ema_50_1w_aligned[i]
        trend_short = price < ema_50_1w_aligned[i]
        
        # ADX filter: strong trend (ADX > 25)
        strong_trend = adx_14_1w_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + uptrend + strong trend
            if price > prev_upper and volume_ok and trend_long and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + downtrend + strong trend
            elif price < prev_lower and volume_ok and trend_short and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian or trend turns bearish or weak trend
            if price < prev_lower or not trend_long or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian or trend turns bullish or weak trend
            if price > prev_upper or not trend_short or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_Volume_ADX_Filter"
timeframe = "1d"
leverage = 1.0