#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 10 or len(df_1d) < 10:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1h Camarilla pivot levels (based on previous day's range)
    # Calculate daily pivot from previous day's OHLC
    # We'll use the previous day's high, low, close to calculate Camarilla levels
    # For simplicity, we'll use rolling window on 1h data to approximate daily OHLC
    # But to be precise, we should get daily OHLC from 1d data
    
    # Get daily OHLC from 1d data and align to 1h
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Align daily OHLC to 1h timeframe
    daily_open_aligned = align_htf_to_ltf(prices, df_1d, daily_open)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate Camarilla levels for each 1h bar using previous day's OHLC
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    # We use previous day's values, so we shift by 1 day (24 hours in 1h data)
    # But since we aligned daily data to 1h, we can use the aligned values directly
    # The aligned daily values represent the value for the entire day, so we use them as is
    
    # Calculate Camarilla levels using previous day's OHLC
    # Shift aligned daily values by 24 to get previous day's values
    prev_daily_open = np.roll(daily_open_aligned, 24)
    prev_daily_high = np.roll(daily_high_aligned, 24)
    prev_daily_low = np.roll(daily_low_aligned, 24)
    prev_daily_close = np.roll(daily_close_aligned, 24)
    
    # Set first 24 values to NaN since we don't have previous day
    prev_daily_open[:24] = np.nan
    prev_daily_high[:24] = np.nan
    prev_daily_low[:24] = np.nan
    prev_daily_close[:24] = np.nan
    
    # Calculate Camarilla R1 and S1
    r1 = prev_daily_close + 1.1 * (prev_daily_high - prev_daily_low) / 12
    s1 = prev_daily_close - 1.1 * (prev_daily_high - prev_daily_low) / 12
    
    # Volume spike detection (2x average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Need enough data for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike in uptrend
            if close[i] > r1[i] and volume[i] > vol_ma_24[i] * 2.0 and \
               ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and \
               ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume spike in downtrend
            elif close[i] < s1[i] and volume[i] > vol_ma_24[i] * 2.0 and \
                 ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and \
                 ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below R1 or trend changes
            if close[i] < r1[i] or ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] or \
               ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above S1 or trend changes
            if close[i] > s1[i] or ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] or \
               ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h/1d trend filter and volume confirmation
# - Camarilla R1/S1 levels provide intraday support/resistance based on previous day's range
# - Breakout above R1 (bullish) or below S1 (bearish) with volume confirmation (2x average)
# - 4h EMA50 and 1d EMA34 trend filters ensure alignment with higher timeframe trends
# - Session filter (08-20 UTC) reduces noise during low-liquidity hours
# - Position size 0.20 limits drawdown and reduces trade frequency
# - Designed to work in both bull and bear markets by following higher timeframe trends
# - Target: 60-150 total trades over 4 years (15-37/year) to avoid excessive fee drag
# - Uses Camarilla pivot levels (proven effective in DB top performers) with proper MTF handling