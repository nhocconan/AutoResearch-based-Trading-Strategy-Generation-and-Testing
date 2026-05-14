#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d timeframe for direction and structure.
# Uses 1d Camarilla pivot levels (S3 and R3) for breakouts with 1d EMA34 trend filter and volume confirmation.
# Designed to work in both bull and bear markets by requiring alignment with daily trend.
# Target: 25-40 trades per year to minimize fee drag and improve generalization.
name = "4h_Camarilla_S3R3_1dEMA34_VolumeBreakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (S3 and R3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Camarilla levels
    # S3 = C - (H - L) * 1.1/2
    # R3 = C + (H - L) * 1.1/2
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 4h timeframe
    s3_4h = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Volume filter: spike above 2.0x 24-period average (1 day of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(s3_4h[i]) or np.isnan(r3_4h[i])):
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
            # Long: price above 1d R3, 1d uptrend (price > EMA34), volume breakout
            if (close[i] > r3_4h[i] and 
                close[i] > ema_34_4h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d S3, 1d downtrend (price < EMA34), volume breakdown
            elif (close[i] < s3_4h[i] and 
                  close[i] < ema_34_4h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 1d S3 or trend reversal
            if close[i] < s3_4h[i] or close[i] < ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 1d R3 or trend reversal
            if close[i] > r3_4h[i] or close[i] > ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals