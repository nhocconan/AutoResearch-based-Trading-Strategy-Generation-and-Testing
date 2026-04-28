#!/usr/bin/env python3
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
    
    # Get daily data for OHLC (used for pivot calculations)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation base
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily OHLC for calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Weekly OHLC for pivot calculation base
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Calculate previous day's range for daily Camarilla
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # Calculate previous week's range for weekly Camarilla
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]
    
    # Daily Camarilla levels (based on previous day)
    daily_range = prev_high_1d - prev_low_1d
    daily_r4 = prev_close_1d + daily_range * 1.1 / 2
    daily_s4 = prev_close_1d - daily_range * 1.1 / 2
    
    # Weekly Camarilla levels (based on previous week)
    weekly_range = prev_high_1w - prev_low_1w
    weekly_r4 = prev_close_1w + weekly_range * 1.1 / 2
    weekly_s4 = prev_close_1w - weekly_range * 1.1 / 2
    
    # Align Daily and Weekly Camarilla levels to 1h timeframe
    daily_r4_aligned = align_htf_to_ltf(prices, df_1d, daily_r4)
    daily_s4_aligned = align_htf_to_ltf(prices, df_1d, daily_s4)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4)
    
    # Weekly trend filter: EMA21 on weekly close
    close_1w_series = pd.Series(close_1w)
    ema21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume filter: above average volume (24-period for 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_r4_aligned[i]) or np.isnan(daily_s4_aligned[i]) or 
            np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below weekly EMA21
        trend_up = close[i] > ema21_1w_aligned[i]
        trend_down = close[i] < ema21_1w_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above BOTH daily R4 and weekly R4 with volume and trend up
        # Short: price breaks below BOTH daily S4 and weekly S4 with volume and trend down
        long_entry = (close[i] > daily_r4_aligned[i]) and (close[i] > weekly_r4_aligned[i]) and vol_filter and trend_up
        short_entry = (close[i] < daily_s4_aligned[i]) and (close[i] < weekly_s4_aligned[i]) and vol_filter and trend_down
        
        # Exit conditions: price returns to opposite daily S4/R4 levels
        long_exit = (close[i] < daily_s4_aligned[i])
        short_exit = (close[i] > daily_r4_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_DailyWeeklyCamarilla_R4S4_WeeklyTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0