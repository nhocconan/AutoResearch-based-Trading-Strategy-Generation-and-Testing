#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for trend and structure
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly EMA10 for trend filter
    close_weekly = df_weekly['close'].values
    ema_10_weekly = pd.Series(close_weekly).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_10_weekly)
    
    # Weekly EMA30 for trend filter
    ema_30_weekly = pd.Series(close_weekly).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_30_weekly)
    
    # Weekly high/low for dynamic range
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    weekly_range = high_weekly - low_weekly
    weekly_range_ma = pd.Series(weekly_range).rolling(window=4, min_periods=4).mean().values
    weekly_range_ma_aligned = align_htf_to_ltf(prices, df_weekly, weekly_range_ma)
    
    # Volume confirmation: volume / 4-period average volume (weekly)
    vol_ma_4 = pd.Series(df_weekly['volume'].values).rolling(window=4, min_periods=4).mean().values
    vol_ratio_weekly = df_weekly['volume'].values / vol_ma_4
    vol_ratio_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vol_ratio_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_10_weekly_aligned[i]) or np.isnan(ema_30_weekly_aligned[i]) or 
            np.isnan(weekly_range_ma_aligned[i]) or np.isnan(vol_ratio_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_fast = ema_10_weekly_aligned[i]
        ema_slow = ema_30_weekly_aligned[i]
        weekly_range_val = weekly_range_ma_aligned[i]
        vol_ratio = vol_ratio_weekly_aligned[i]
        
        if position == 0:
            # Enter long: fast EMA above slow EMA, price above weekly midpoint, volume spike
            weekly_mid = (high_weekly[i] + low_weekly[i]) / 2 if not (np.isnan(high_weekly[i]) or np.isnan(low_weekly[i])) else 0
            if (ema_fast > ema_slow and 
                price_close > weekly_mid and 
                vol_ratio > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: fast EMA below slow EMA, price below weekly midpoint, volume spike
            elif (ema_fast < ema_slow and 
                  price_close < weekly_mid and 
                  vol_ratio > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: EMA cross in opposite direction or low volume
            if position == 1 and (ema_fast < ema_slow or vol_ratio < 0.8):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (ema_fast > ema_slow or vol_ratio < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyEMA10_30_Volume_Momentum"
timeframe = "6h"
leverage = 1.0