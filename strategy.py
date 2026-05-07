#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_S1R1_Breakout_Trend"
timeframe = "1h"
leverage = 1.0

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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h data for Camarilla pivot levels (from previous 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels from previous 4h bar
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3
    range_4h = prev_high_4h - prev_low_4h
    
    # Camarilla S1 and R1 levels (most significant)
    s1_4h = prev_close_4h - (range_4h * 1.08 / 2)
    r1_4h = prev_close_4h + (range_4h * 1.08 / 2)
    
    # Align 4h levels to 1h timeframe
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    
    # Volume spike detection (20-period average for 1h = ~3.3 hours)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0 and in_session:
            # Long: price above S1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s1_4h_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price below R1 with volume and daily downtrend
            elif close[i] < r1_4h_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops or outside session
            if (close[i] < s1_4h_aligned[i] or 
                volume[i] < vol_ma_20[i] * 1.3 or 
                not in_session):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above R1 or volume drops or outside session
            if (close[i] > r1_4h_aligned[i] or 
                volume[i] < vol_ma_20[i] * 1.3 or 
                not in_session):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla S1/R1 breakout with daily trend filter and volume confirmation
# - Uses 4h Camarilla S1/R1 levels from previous 4h bar as support/resistance
# - Entry only during 08-20 UTC session to avoid low-liquidity hours
# - Long when price breaks above S1 with volume spike (2x avg) in daily uptrend
# - Short when price breaks below R1 with volume spike in daily downtrend
# - Exit when price returns to S1/R1, volume weakens, or outside session
# - Position size 0.20 limits risk per trade
# - Designed for 1h timeframe with 15-37 trades/year target to avoid fee drag
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend) markets