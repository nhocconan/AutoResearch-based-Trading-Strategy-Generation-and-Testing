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
    
    # Load daily data for Donchian(20) and ATR(14) - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) channels
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    upper_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close, 1))
    tr3 = np.abs(low_daily - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Donchian channels and ATR to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_daily, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_daily, lower_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_daily, atr_14)
    
    # Load weekly data for ATR-based volatility regime filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly ATR(14)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    tr1_w = high_weekly - low_weekly
    tr2_w = np.abs(high_weekly - np.roll(close_weekly, 1))
    tr3_w = np.abs(low_weekly - np.roll(close_weekly, 1))
    tr1_w[0] = tr2_w[0] = tr3_w[0] = 0
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_weekly = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio: weekly ATR / daily ATR (volatility regime)
    atr_ratio = atr_weekly / np.roll(atr_14, 24)  # Approximate weekly ATR from daily (14*24 hours)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_weekly, atr_ratio)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
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
            # Long: Price breaks above upper Donchian(20) AND high volatility regime (expanding ATR)
            if (close[i] > upper_20_aligned[i] and 
                atr_ratio_aligned[i] > 1.2):  # Volatility expansion regime
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(20) AND high volatility regime
            elif (close[i] < lower_20_aligned[i] and 
                  atr_ratio_aligned[i] > 1.2):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel OR volatility contracts
            if position == 1:
                if (close[i] < lower_20_aligned[i] or 
                    atr_ratio_aligned[i] < 0.8):  # Volatility contraction
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > upper_20_aligned[i] or 
                    atr_ratio_aligned[i] < 0.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Donchian20_VolatilityRegime_Expansion"
timeframe = "12h"
leverage = 1.0