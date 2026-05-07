#!/usr/bin/env python3
# 12h_WeeklyPivot_CamR3S3_Breakout_1dTrend_Volume
# Hypothesis: Weekly Pivot + Daily Camarilla R3/S3 breakouts on 12h timeframe
# Weekly pivot provides institutional support/resistance from weekly structure.
# Daily Camarilla R3/S3 acts as intraday breakout levels within the week.
# Combined with 1d EMA34 trend filter and volume confirmation to avoid false breakouts.
# Works in bull markets via long breakouts above weekly pivot + daily R3,
# and in bear markets via short breakdowns below weekly pivot + daily S3.
# Target: 15-35 trades per year (~60-140 over 4 years) with position size 0.25.

name = "12h_WeeklyPivot_CamR3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Load daily data for Camarilla and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate Weekly Pivot Points (using prior week's OHLC)
    # Standard pivot: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # We use R1 and S1 as key levels
    weekly_high = df_weekly['high'].shift(1).values  # Previous week's high
    weekly_low = df_weekly['low'].shift(1).values    # Previous week's low
    weekly_close = df_weekly['close'].shift(1).values # Previous week's close
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 12h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Calculate Daily Camarilla R3 and S3 levels (using prior day's OHLC)
    # Camarilla: R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    daily_high = df_daily['high'].shift(1).values    # Previous day's high
    daily_low = df_daily['low'].shift(1).values      # Previous day's low
    daily_close = df_daily['close'].shift(1).values  # Previous day's close
    
    camarilla_r3 = daily_close + 1.1 * (daily_high - daily_low) / 2
    camarilla_s3 = daily_close - 1.1 * (daily_high - daily_low) / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_daily['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_daily, ema_34_1d)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above weekly R1 + daily R3 OR below weekly S1 + daily S3
        breakout_up = close[i] > weekly_r1_aligned[i] and close[i] > camarilla_r3_aligned[i]
        breakout_down = close[i] < weekly_s1_aligned[i] and close[i] < camarilla_s3_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: upward breakout above weekly R1 AND daily R3 + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout below weekly S1 AND daily S3 + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks back below weekly R1 OR daily R3 (failed breakout) or trend reversal
            if close[i] < weekly_r1_aligned[i] or close[i] < camarilla_r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks back above weekly S1 OR daily S3 (failed breakdown) or trend reversal
            if close[i] > weekly_s1_aligned[i] or close[i] > camarilla_s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals