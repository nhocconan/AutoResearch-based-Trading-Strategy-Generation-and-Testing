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
    
    # Load weekly and daily data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    df_daily = get_htf_data(prices, '1d')
    
    if len(df_weekly) < 5 or len(df_daily) < 10:
        return np.zeros(n)
    
    # Weekly trend: EMA(5) vs EMA(20) on weekly close
    weekly_close = df_weekly['close'].values
    weekly_ema5 = pd.Series(weekly_close).ewm(span=5, adjust=False, min_periods=5).mean().values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend = weekly_ema5 > weekly_ema20  # True = bullish, False = bearish
    
    # Align weekly trend to 6h
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend.astype(float))
    
    # Daily ATR(14) for volatility filter
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ATR to 6h
    atr14_aligned = align_htf_to_ltf(prices, df_daily, atr14)
    
    # 6h EMA(20) for dynamic trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(weekly_trend_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(ema20[i]) or np.isnan(vol_avg_20[i])):
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
            # Long conditions: weekly bullish + price above EMA20 + volume surge
            if (weekly_trend_aligned[i] > 0.5 and 
                close[i] > ema20[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: weekly bearish + price below EMA20 + volume surge
            elif (weekly_trend_aligned[i] < 0.5 and 
                  close[i] < ema20[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: weekly trend reversal or price crosses EMA20 in opposite direction
            if position == 1:
                if (weekly_trend_aligned[i] < 0.5 or close[i] < ema20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (weekly_trend_aligned[i] > 0.5 or close[i] > ema20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyTrend_EMA20_Volume"
timeframe = "6h"
leverage = 1.0