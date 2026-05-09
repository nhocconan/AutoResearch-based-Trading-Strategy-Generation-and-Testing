#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w timeframe for direction and structure.
# Uses 1w Camarilla pivot levels (S2 and R2) for breakouts with 1w EMA20 trend filter and volume confirmation.
# Designed to work in both bull and bear markets by requiring alignment with weekly trend.
# Target: 15-30 trades per year to minimize fee drag and improve generalization.
name = "12h_Camarilla_S2R2_1wEMA20_VolumeBreakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot levels and EMA20
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1w Camarilla pivot levels (S2 and R2)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Camarilla levels
    # S2 = C - (H - L) * 1.1/4
    # R2 = C + (H - L) * 1.1/4
    s2_1w = close_1w - (high_1w - low_1w) * 1.1 / 4.0
    r2_1w = close_1w + (high_1w - low_1w) * 1.1 / 4.0
    
    # Align 1w Camarilla levels to 12h timeframe
    s2_12h = align_htf_to_ltf(prices, df_1w, s2_1w)
    r2_12h = align_htf_to_ltf(prices, df_1w, r2_1w)
    
    # Volume filter: spike above 2.0x 12-period average (1 day of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 12)  # Wait for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_12h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(s2_12h[i]) or np.isnan(r2_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade during active hours (8 AM - 8 PM UTC) to avoid low liquidity
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above 1w R2, 1w uptrend (price > EMA20), volume breakout
            if (close[i] > r2_12h[i] and 
                close[i] > ema_20_12h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below 1w S2, 1w downtrend (price < EMA20), volume breakdown
            elif (close[i] < s2_12h[i] and 
                  close[i] < ema_20_12h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 1w S2 or trend reversal
            if close[i] < s2_12h[i] or close[i] < ema_20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 1w R2 or trend reversal
            if close[i] > r2_12h[i] or close[i] > ema_20_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals