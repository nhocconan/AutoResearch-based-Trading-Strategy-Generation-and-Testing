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
    
    # Load weekly data for ATR(14) - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly ATR(14)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    tr1 = high_w - low_w
    tr2 = np.abs(high_w - np.roll(close_w, 1))
    tr3 = np.abs(low_w - np.roll(close_w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_weekly, atr_14)
    
    # Load daily data for Donchian(20) and EMA(50) - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) channels
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    upper_20 = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    upper_20_aligned = align_htf_to_ltf(prices, df_daily, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_daily, lower_20)
    
    # Calculate daily EMA(50)
    ema_50 = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_daily, ema_50)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above upper Donchian(20) with volume and above EMA50
            if (close[i] > upper_20_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i] and
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(20) with volume and below EMA50
            elif (close[i] < lower_20_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i] and
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: ATR-based trailing stop
            if position == 1:
                # Long: stop if price drops below highest high since entry minus 2*ATR
                # Simplified: exit if price < EMA50 or price drops significantly
                if close[i] < ema_50_aligned[i] or close[i] < (upper_20_aligned[i] - 2.0 * atr_14_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short: stop if price rises above lowest low since entry plus 2*ATR
                # Simplified: exit if price > EMA50 or price rises significantly
                if close[i] > ema_50_aligned[i] or close[i] > (lower_20_aligned[i] + 2.0 * atr_14_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_WeeklyATR_DailyDonchian_EMA50_Volume"
timeframe = "12h"
leverage = 1.0