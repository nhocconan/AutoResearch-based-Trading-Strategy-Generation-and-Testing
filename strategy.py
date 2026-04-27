#!/usr/bin/env python3
"""
1d_Weekly_Trend_Follower_v2
Hypothesis: Uses weekly Donchian channel breakout with ADX trend filter and volume confirmation on daily timeframe.
Designed for low trade frequency (10-25 trades/year) to minimize fee decay, capturing major trends in both bull and bear markets.
Weekly trend filter prevents counter-trend entries, while daily entry timing improves precision.
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channel (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Daily ADX for trend strength (14-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate +DM and -DM
    up_move = daily_high - np.roll(daily_high, 1)
    down_move = np.roll(daily_low, 1) - daily_low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        adx_val = adx_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Only trade when trend is strong (ADX > 25)
        if adx_val > 25:
            if position == 0:
                # Long: price breaks above weekly Donchian high with volume
                if close[i] > donchian_high_val and vol_conf:
                    signals[i] = size
                    position = 1
                # Short: price breaks below weekly Donchian low with volume
                elif close[i] < donchian_low_val and vol_conf:
                    signals[i] = -size
                    position = -1
            elif position == 1:
                # Exit long: price breaks below weekly Donchian low
                if close[i] < donchian_low_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            elif position == -1:
                # Exit short: price breaks above weekly Donchian high
                if close[i] > donchian_high_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
        else:
            # Weak trend - stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "1d_Weekly_Trend_Follower_v2"
timeframe = "1d"
leverage = 1.0