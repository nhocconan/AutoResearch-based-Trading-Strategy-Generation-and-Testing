#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA Trend + Weekly Bollinger Band Breakout with Volume Confirmation
# Uses KAMA (Kaufman Adaptive Moving Average) on daily timeframe for trend direction,
# combined with weekly Bollinger Band upper/lower breakouts for entry signals.
# Volume confirmation filters out false breakouts. Designed to work in both bull and bear markets
# by adapting to volatility and using multiple timeframe confirmation.
# Target: 20-30 trades/year (80-120 over 4 years) to minimize fee drag.
name = "1d_KAMA_Trend_WeeklyBB_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Get daily data for KAMA calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20, 2)
    weekly_close = df_weekly['close'].values
    weekly_ma = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    weekly_std = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    weekly_upper = weekly_ma + 2 * weekly_std
    weekly_lower = weekly_ma - 2 * weekly_std
    
    # Align weekly Bollinger Bands to daily
    upper_aligned = align_htf_to_ltf(prices, df_weekly, weekly_upper)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, weekly_lower)
    ma_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ma)
    
    # Calculate KAMA on daily close
    # Efficiency ratio (ER) over 10 periods
    change = np.abs(np.diff(df_daily['close'], prepend=df_daily['close'].iloc[0]))
    volatility = np.abs(np.diff(df_daily['close'])).rolling(window=10, min_periods=10).sum().values
    volatility = np.concatenate([[np.nan], volatility[:-1]])  # align with change
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.full_like(df_daily['close'], np.nan, dtype=float)
    kama[0] = df_daily['close'].iloc[0]
    for i in range(1, len(df_daily)):
        kama[i] = kama[i-1] + sc[i] * (df_daily['close'].iloc[i] - kama[i-1])
    kama_values = kama
    
    # Align KAMA to daily (same timeframe, no alignment needed but keep for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama_values)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(kama_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: Close above weekly upper Bollinger Band with KAMA uptrend and volume spike
            if close[i] > upper_aligned[i] and close[i] > kama_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close below weekly lower Bollinger Band with KAMA downtrend and volume spike
            elif close[i] < lower_aligned[i] and close[i] < kama_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below weekly middle band OR KAMA turns down
            if close[i] < ma_aligned[i] or close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above weekly middle band OR KAMA turns up
            if close[i] > ma_aligned[i] or close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals