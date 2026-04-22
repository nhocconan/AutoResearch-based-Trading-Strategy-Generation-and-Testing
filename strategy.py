# 6H_WeeklyPivot_Donchian20_Trend_Volume
# Hypothesis: 6h Donchian(20) breakout in direction of weekly pivot (from prior week) with volume confirmation.
# Uses weekly pivot as long-term bias filter, Donchian breakout for entry timing, volume for momentum confirmation.
# Works in bull/bear by aligning with weekly structure. Target: 15-30 trades/year (60-120 total over 4 years).
# Designed to avoid overtrading with strict entry conditions.

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
    
    # Load weekly data for pivot calculation - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (based on previous weekly bar)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly_prev = df_weekly['close'].values
    
    # Standard pivot point: P = (H + L + C) / 3
    # Support 1: S1 = (2 * P) - H
    # Resistance 1: R1 = (2 * P) - L
    pp = (high_weekly + low_weekly + close_weekly_prev) / 3.0
    r1 = (2 * pp) - high_weekly
    s1 = (2 * pp) - low_weekly
    
    # Align weekly pivot levels to 6h timeframe (previous week's levels)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Load daily data for trend filter (EMA34)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema_34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_34_daily)
    
    # Calculate 6h Donchian channels (20-period)
    # Using 60-period lookback for 6h (since 20 * 6h = 5 days, but we use fixed 20 periods)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_34_daily_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above resistance with weekly bullish bias and volume
            if (high[i] > highest_high[i] and 
                close[i] > pp_aligned[i] and  # Above weekly pivot = bullish bias
                close[i] > ema_34_daily_aligned[i] and  # Daily uptrend
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below support with weekly bearish bias and volume
            elif (low[i] < lowest_low[i] and 
                  close[i] < pp_aligned[i] and  # Below weekly pivot = bearish bias
                  close[i] < ema_34_daily_aligned[i] and  # Daily downtrend
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to weekly pivot point
            if position == 1:
                if close[i] <= pp_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= pp_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyPivot_Donchian20_Trend_Volume"
timeframe = "6h"
leverage = 1.0