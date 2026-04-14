#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Choppiness Index regime filter + 1-week EMA trend + volume spike
# Long when: Choppiness > 61.8 (range), price > weekly EMA (uptrend), volume > 2x average
# Short when: Choppiness > 61.8 (range), price < weekly EMA (downtrend), volume > 2x average
# Exit when Choppiness < 38.2 (trending regime) or opposite signal
# Uses weekly EMA to avoid counter-trend trades and Choppiness to identify range-bound markets
# Target: 15-30 trades per symbol over 4 years (4-7.5/year) to minimize fee drag
# This combines regime detection with trend following for range-bound markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index (14-period) on daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily[1:] - low_daily[1:]
    tr2 = np.abs(high_daily[1:] - close_daily[:-1])
    tr3 = np.abs(low_daily[1:] - close_daily[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original array
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_daily).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_daily).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (atr * 14)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # Avoid division by zero
    
    # Calculate weekly EMA for trend filter (21-period)
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_ma_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for 14-period chop + 20-period vol
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(vol_ma_daily_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_daily_current = volume[i]  # Current daily volume
        
        if position == 0:
            # Long setup: range market (chop > 61.8), price above weekly EMA, volume spike
            if (chop_aligned[i] > 61.8 and 
                price > ema_weekly_aligned[i] and 
                vol_daily_current > 2.0 * vol_ma_daily_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: range market (chop > 61.8), price below weekly EMA, volume spike
            elif (chop_aligned[i] > 61.8 and 
                  price < ema_weekly_aligned[i] and 
                  vol_daily_current > 2.0 * vol_ma_daily_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trending regime (chop < 38.2) or price crosses below weekly EMA
            if chop_aligned[i] < 38.2 or price < ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trending regime (chop < 38.2) or price crosses above weekly EMA
            if chop_aligned[i] < 38.2 or price > ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Choppiness_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0