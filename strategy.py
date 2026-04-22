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
    
    # Load weekly data for trend - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend
    weekly_close = df_weekly['close'].values
    ema_21_weekly = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Load daily data for daily pivot levels - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot points
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    daily_r1 = 2 * daily_pivot - daily_low
    daily_s1 = 2 * daily_pivot - daily_high
    daily_r2 = daily_pivot + daily_range
    daily_s2 = daily_pivot - daily_range
    
    # Align weekly trend and daily pivot levels to 6h timeframe
    ema_21_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_21_weekly)
    daily_pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    daily_r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    daily_r2_aligned = align_htf_to_ltf(prices, df_daily, daily_r2)
    daily_s2_aligned = align_htf_to_ltf(prices, df_daily, daily_s2)
    
    # Calculate 6h ADX(14) for trend strength
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_21_weekly_aligned[i]) or np.isnan(daily_pivot_aligned[i]) or 
            np.isnan(daily_r1_aligned[i]) or np.isnan(daily_s1_aligned[i]) or
            np.isnan(daily_r2_aligned[i]) or np.isnan(daily_s2_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly uptrend + price breaks above daily R2 with volume + strong trend
            if (close[i] > ema_21_weekly_aligned[i] and  # Weekly uptrend
                close[i] > daily_r2_aligned[i] and       # Break above daily R2
                volume[i] > 2.0 * vol_avg_20[i] and    # Volume spike
                adx[i] > 25):                          # Strong trend
                signals[i] = 0.25
                position = 1
            # Short: Weekly downtrend + price breaks below daily S2 with volume + strong trend
            elif (close[i] < ema_21_weekly_aligned[i] and  # Weekly downtrend
                  close[i] < daily_s2_aligned[i] and       # Break below daily S2
                  volume[i] > 2.0 * vol_avg_20[i] and    # Volume spike
                  adx[i] > 25):                          # Strong trend
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to daily pivot level
            if position == 1:
                if close[i] < daily_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > daily_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyTrend_DailyPivot_R2S2_Breakout"
timeframe = "6h"
leverage = 1.0