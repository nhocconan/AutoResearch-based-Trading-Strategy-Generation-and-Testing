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
    
    # Load weekly data for EMA(50) - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) from weekly close
    weekly_close = df_weekly['close'].values
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA(50) to daily timeframe
    weekly_ema_50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_50)
    
    # Load daily data for ATR(14) - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR(14) to daily timeframe (no shift needed as it's same TF)
    atr_14_aligned = atr_14  # Already daily
    
    # Calculate daily volume average (20-period)
    daily_volume = df_daily['volume'].values
    vol_avg_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(weekly_ema_50_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
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
            # Long: Price above weekly EMA(50) with volume confirmation
            if (close[i] > weekly_ema_50_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA(50) with volume confirmation
            elif (close[i] < weekly_ema_50_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses weekly EMA(50) in opposite direction
            if position == 1:
                if close[i] < weekly_ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > weekly_ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_WeeklyEMA50_Volume_Session"
timeframe = "1d"
leverage = 1.0