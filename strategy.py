#!/usr/bin/env python3
"""
12h_WeeklyDonchian_Breakout_DailyTrend_VolumeFilter
Hypothesis: Breakout of weekly Donchian(20) channels on 12h timeframe, filtered by daily EMA50 trend and volume > 1.5x median.
Exit on opposite Donchian touch or ATR(14) stoploss. Designed for 12h to capture multi-day trends while avoiding overtrading.
Target: 50-150 total trades over 4 years (12-37/year). Works in both bull and bear via trend filter and volatility-based stops.
"""

name = "12h_WeeklyDonchian_Breakout_DailyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- Weekly Donchian Channels (20-period) ---
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate rolling high/low
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # --- Daily Trend Filter: EMA50 ---
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # --- Volume Filter: above 1.5x median of last 30 periods ---
    vol_median = pd.Series(volume_12h).rolling(window=30, min_periods=15).median().values
    vol_threshold = vol_median * 1.5
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period (max of weekly lookback and daily EMA)
    start_idx = 50  # for weekly Donchian and daily EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_12h[i] <= entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine daily trend
        trend_up = close_12h[i] > ema50_daily_aligned[i]
        trend_down = close_12h[i] < ema50_daily_aligned[i]
        
        # Volume filter: above 1.5x median
        vol_ok = volume_12h[i] > vol_threshold[i]
        
        if position == 0:
            # Look for entries only in direction of daily trend with volume
            if close_12h[i] > donchian_high_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above weekly Donchian high + daily uptrend + volume
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif close_12h[i] < donchian_low_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below weekly Donchian low + daily downtrend + volume
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_12h[i] <= entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses below weekly Donchian low
                elif close_12h[i] <= donchian_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Stoploss
                if close_12h[i] >= entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: price touches or crosses above weekly Donchian high
                elif close_12h[i] >= donchian_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals