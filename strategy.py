#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (standard formula)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Pivot Point = (H + L + C) / 3
    pp = (daily_high + daily_low + daily_close) / 3.0
    # Resistance 3 = H + 2*(PP - L)
    r3 = daily_high + 2 * (pp - daily_low)
    # Support 3 = L - 2*(H - PP)
    s3 = daily_low - 2 * (daily_high - pp)
    
    # Align daily Camarilla levels to 4h timeframe (with 1-bar delay for completed daily bar)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: spike above 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r3_4h[i]) or np.isnan(s3_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Trade only during active hours: 8 AM - 8 PM UTC
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price above daily R3, 1d uptrend (price > EMA50), volume breakout
            if (close[i] > r3_4h[i] and 
                close[i] > ema_50_4h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.30
                position = 1
            # Short: price below daily S3, 1d downtrend (price < EMA50), volume breakdown
            elif (close[i] < s3_4h[i] and 
                  close[i] < ema_50_4h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price below daily S3 or trend reversal
            if close[i] < s3_4h[i] or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price above daily R3 or trend reversal
            if close[i] > r3_4h[i] or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals