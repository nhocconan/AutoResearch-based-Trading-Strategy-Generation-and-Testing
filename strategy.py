#!/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with 1w EMA Trend Filter and Volume Confirmation
Long when price touches Camarilla S1/S2 in uptrend (price > 1w EMA50) with volume spike.
Short when price touches Camarilla R1/R2 in downtrend (price < 1w EMA50) with volume spike.
Exit when price reaches opposite Camarilla level or trend reverses.
Designed for low trade frequency (12-30/year) to minimize fee drift.
"""
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
    
    # Load weekly data for EMA trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    ema50_weekly = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Load daily data for Camarilla pivots
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # H, L, C from previous daily bar
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    # Camarilla equations
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    R2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    S2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 12h
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1)
    R2_aligned = align_htf_to_ltf(prices, df_daily, R2)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1)
    S2_aligned = align_htf_to_ltf(prices, df_daily, S2)
    
    # Calculate 12h volume average (24-period = 12 days)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after volume MA warmup
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(S2_aligned[i]) or np.isnan(vol_avg_24[i])):
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
            # Long: Price touches S1/S2 in uptrend with volume spike
            if (ema50_aligned[i] > 0 and  # Valid EMA
                ((low[i] <= S1_aligned[i] and high[i] >= S1_aligned[i]) or
                 (low[i] <= S2_aligned[i] and high[i] >= S2_aligned[i])) and
                close[i] > ema50_aligned[i] and  # Uptrend filter
                volume[i] > 2.0 * vol_avg_24[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price touches R1/R2 in downtrend with volume spike
            elif (ema50_aligned[i] > 0 and  # Valid EMA
                  ((high[i] >= R1_aligned[i] and low[i] <= R1_aligned[i]) or
                   (high[i] >= R2_aligned[i] and low[i] <= R2_aligned[i])) and
                  close[i] < ema50_aligned[i] and  # Downtrend filter
                  volume[i] > 2.0 * vol_avg_24[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches R1 or trend turns down
                if high[i] >= R1_aligned[i] or close[i] < ema50_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches S1 or trend turns up
                if low[i] <= S1_aligned[i] or close[i] > ema50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_CamarillaPivotReversal_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0
#%%